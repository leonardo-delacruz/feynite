(* ::Package:: *)

(*TENSOR INTEGRANDS*)
(*ADJUGATE*)
adjA[x__]:=Simplify[Inverse[x]Det[x]];
(*B matrix is required to obtain parametric representation of a tensor integral*)
(*Wick contractions to obtain parametric representation*)
wick2[X__]:=(Block[{list=X},Sum[WC[{list[[1]],list[[n]]}] WickAux[Drop[Drop[list,{n}],1]],{n,2,Length[list]}]])//.WickAux->wick2;
wick2[{x_,y_}]:=WC[{x,y}];

Module[{idx=0},Paux[i_,\[Mu]_]:=(idx+=1; adj[i,Subscript[s,idx]] B[Subscript[s,idx],\[Mu]])];

genTensorIntegrand[r_]:=Module[{Z0,allTensors,nonVanish,allWIcks,toIntegrate,resIntegrated,factor},
Z0=(I T)^(-d L/2)Uc^(-d/2);
factor=(Fc^(d L/2-Nn)Uc^(-d/2-d L/2+Nn-r) I^(-2 Nn));
allTensors=(-1)^r  sW[{}]Product[lV[k[\[Sigma][i]],\[Mu][i]]+Uc^-1lV[-1P[\[Sigma][i]],\[Mu][i]],{i,1,r}]//Expand;
nonVanish=((DeleteCases[allTensors/.Plus->List,-X__])//.{lV[k[i_],\[Mu]_]sW[X__]:>sW[Append[X,k[i,\[Mu]]]]})/.{sW[{}]->1};
allWIcks=(nonVanish/.sW->wick2);
toIntegrate=(1/factor Z0 I^(-Nn) T^(Nn-1)Exp[-I T Fc/Uc](allWIcks/.{WC[{k[\[Sigma]s_,\[Mu]_],k[\[Rho]_,\[Nu]_]}]:>I/(2T)Uc^-1 adj[\[Sigma]s,\[Rho]]MT[\[Mu],\[Nu]]}))//Simplify//PowerExpand;
resIntegrated=Table[(I Fc/Uc)^(-1-Exponent[toIntegrate[[i]],T])Gamma[1+Exponent[toIntegrate[[i]],T]](toIntegrate[[i]]/.{E^(-((I Fc T)/Uc)) ->1,T->1})//PowerExpand,{i,1,Length[toIntegrate]}]//Simplify;
(*USES GAMMA PROPERTIES to produce a numerator with a simple Gamma functions*)
If[r==0,{{Gamma[-d L/2+Nn]} ,factor},{If[r==1,resIntegrated,resIntegrated//.{Gamma[n_+x_]/;n>-Floor[r/2]:>(n-1+x)Gamma[n-1+x],Gamma[-((d L)/2)+Nn]:>(-((d L)/2)+Nn-1)Gamma[-((d L)/2)+Nn-1]}],factor}](*/.{lV[P[\[Sigma]s_],\[Mu]_]:>Paux[\[Sigma]s,\[Mu]]}*)
]

Bmat2[r_][ufdata__]:=ufdata[[3]]/.{\[Mu]->r};
cP3[\[Sigma]_,\[Rho]_,datas__]:=Module[{bmat,adjugates,dens,ufdata},
dens=datas[[1]];
ufdata=datas[[2]];
adjugates=datas[[3]];
bmat=Bmat2[\[Rho]][ufdata];
Sum[((adjugates)[[\[Sigma]]][[i]]bmat)[[i]],{i,1,Length[dens[[3]]]}]//Expand
]

(*Auxiliary function to deal with uncontracted indices: creates unique indices*)
index[x_,y_]:=Module[{li1,li2},
li1=\[Mu][Unique[]];
li2=\[Mu][Unique[]];
mt[li1,li2]sV[x,li1]sV[y,li2]
]
(*uncontracts only loop momenta: use auxiliary vector \[Lambda] to extract indices*)
unContractnumerator[num_]:=Module[{num1,num2},
num1=num//.{Power[dL[x_,y_],n_Integer]:>Apply[Times,Table[index[x,y],{o,1,n}]]};
num2=(num1//.{dL[x_,y_]:>index[x,y]})/.{sV[k[i_],\[Mu][j_]]:> \[Lambda][i,j]};
Return[num2/.{mt->MT,sV->lV}]
]

(*uncontracted numerator as list*)
unContractnumeratorList[num_]:=Module[{listnums},
listnums=If[Head[num]===Plus,List@@(unContractnumerator[num//Expand]//Expand),{unContractnumerator[num//Expand]}];
If[num===1,{{1,1}},Table[{listnums[[i]]/.{\[Lambda][z_,x_]:>1},Coefficient[listnums[[i]],listnums[[i]]/.{\[Lambda][z_,x_]:>1}]},{i,1,Length[listnums]}]]
]
	
(*returns the numerator data, loop labels and tensors*)
toDataNum[num_]:=Module[{listTerms,indices,indexRules,dataNum},
listTerms=unContractnumeratorList[num];
indices=Table[If[Depth[listTerms[[i]][[2]]]<=2,{listTerms[[i]][[2]]/.{\[Lambda][r__,z_]:>\[Mu][z]}},List@@(listTerms[[i]][[2]])/.{\[Lambda][r__,z_]:>\[Mu][z]}],{i,1,Length[listTerms]}];
indexRules=Table[If[Variables[indices[[i]]]=={},{},MapThread[Rule,{indices[[i]],Table[\[Mu][j],{j,1,Length[indices[[i]]]}]}]],{i,1,Length[indices]}];
(*working function*)
dataNum={Table[listTerms[[i]][[1]]/.indexRules[[i]],{i,1,Length[listTerms]}],
Table[If[Depth[listTerms[[i]][[2]]]==1,1,If[1<Depth[listTerms[[i]][[2]]]<=2,{listTerms[[i]][[2]]}/.{\[Lambda][i_,x_]:>i},List@@(listTerms[[i]][[2]])/.{\[Lambda][i_,x_]:>i}]],{i,1,Length[listTerms]}]};
If[num==1,dataNum={{1},{{}}},dataNum];
Return[dataNum]
]


(*Just for one term*)
numSingleTerm[momentaNumerator__,setLoop__,datas__,highestRank_,edges_,tensorsuptoHighest__,nprops_,loops__,kin__]:=Module[{adjugates,ufdata,indices,gentensors,tensorStructures,tensors,singleTermData,preptoSumArrays,numArraySingleTerm},
ufdata=datas[[2]];
adjugates=datas[[3]];
indices=Table[Rule[\[Sigma][i],setLoop[[i]]],{i,1,Length[setLoop]}];
gentensors=tensorsuptoHighest[[Length[setLoop]+1]];
tensorStructures=(gentensors/.indices)/.{lV[P[in_],m_]:>cP3[in,m,datas],adj[e1_,e2_]:>(adjugates)[[e1]][[e2]],Nn->nprops,L->loops};
tensors=tensorStructures//.Gamma[n_+x__]/;n>(nprops-Floor[highestRank/2]):>(n-1+x)Gamma[n-1+x];
singleTermData={Uc^(highestRank-Length[setLoop])(tensors[[1]]*momentaNumerator)//.kin,Uc^(-(highestRank-Length[setLoop]))tensors[[2]]}/.z[edges]->1;
preptoSumArrays=Join[(singleTermData[[1]]/.{Uc->ufdata[[1]],Fc->ufdata[[2]]}),If[highestRank==0,{},ConstantArray[0,Length[tensorsuptoHighest[[highestRank+1]][[1]]]-Length[gentensors[[1]]]]]]/.{Gamma[x_]:>1}
]

allTermsNum[dataNum__,datas__,highestRank_,edges_,tensorsuptoHighest__,nprops_,loops__,kin__]:=Module[{tensorsTogether,ufdata},
ufdata=datas[[2]]/.z[edges]->1;
tensorsTogether=Plus@@(Parallelize[numSingleTerm[dataNum[[1]][[#]],dataNum[[2]][[#]],datas,highestRank,edges,tensorsuptoHighest,nprops,loops,kin]&/@Range[Length[dataNum[[1]]]]]);
Return[CoefficientArrays[((Plus@@tensorsTogether)),Array[z,edges-1]]]
]


arrayIntegrandData[num_,dens__,kin__]:=Module[{dataNum,edges,nprops,ufdata,loops,mat,adjugates,highestRank,tensorsuptoHighest,datas},
dataNum=toDataNum[num];
edges=Length[dens[[1]]];
nprops=Length[dens[[1]]];
ufdata=((UFdata[dens]/.z[edges]->1)//.kin)//Expand;
loops=Length[dens[[3]]];
mat=ufdata[[4]];
adjugates=adjA[mat];
highestRank=Max[Length[#]&/@(dataNum[[2]])];
tensorsuptoHighest=Table[genTensorIntegrand[i],{i,0,highestRank}];
datas={dens,ufdata,adjugates};
Return[allTermsNum[dataNum,datas,highestRank,edges,tensorsuptoHighest,nprops,loops,kin]]
]

arrayIntegrandDataNoGauge[num_,dens__,kin__]:=Module[{dataNum,edges,nprops,ufdata,loops,mat,adjugates,highestRank,tensorsuptoHighest,datas},
dataNum=toDataNum[num];
edges=Length[dens[[1]]];
nprops=Length[dens[[1]]];
ufdata=((UFdata[dens])//.kin)//Expand;
loops=Length[dens[[3]]];
mat=ufdata[[4]];
adjugates=adjA[mat];
highestRank=Max[Length[#]&/@(dataNum[[2]])];
tensorsuptoHighest=Table[genTensorIntegrand[i],{i,0,highestRank}];
datas={dens,ufdata,adjugates};
Return[allTermsNum[dataNum,datas,highestRank,edges,tensorsuptoHighest,nprops,loops,kin]]
]

