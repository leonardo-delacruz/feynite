(* ::Package:: *)

(*Lorentz Vectors*)
lV[sum_Plus,m_]:=lV[#,m]&/@sum;
lV[Times[a_Integer ,A_],m_]:=a*lV[A,m];
lV /: lV[i_,\[Mu]_]^2:= dL[i,i];
lV /: lV[i_,\[Mu]_] lV[j_,\[Mu]_]:= dL[i,j];

(*dot Lorentz*)
SetAttributes[dL,Orderless]
dL[sum_Plus,A__]:=dL[#,A]&/@sum;
dL[A__,sum_Plus]:=dL[A,#]&/@sum;
dL[Times[a_Integer ,A__ ],X__]:=a*dL[A,X];
dL[X__,Times[a_Integer ,A__ ]]:=a*dL[X,A];
dL[x_]:=dL[x,x];
dL[0,x_]:=0;

(*metric tensor*)
SetAttributes[MT, Orderless]
MT[\[Mu]_,\[Mu]_]:= d;
MT /: MT[\[Mu]_,\[Mu]p_]^2:= d;
MT /: MT[\[Mu]_,\[Nu]_] lV[i_,\[Nu]_]:= lV[i,\[Mu]];
MT /: MT[\[Mu]_,\[Nu]_] lV[i_,\[Mu]_]:= lV[i,\[Nu]];
MT /: MT[\[Mu]_,\[Nu]_]MT[\[Mu]_,\[Rho]_]:= MT[\[Nu],\[Rho]];
MT /: MT[\[Mu]_,\[Nu]_]MT[\[Rho]_,\[Nu]_]:= MT[\[Mu],\[Rho]];


monomialList[poly_,vars_]:=Times@@(vars^#)&/@CoefficientRules[poly,vars,"Lexicographic"][[All,1]];
monomialList2[poly_,vars__]:=Last@*FactorTermsList/@MonomialList[poly,vars,"NegativeDegreeLexicographic"]

(*BASED ON MINT by Roman Lee*)
UFdata[{momprop__,masses__,loopMom__}]:=Module[{t1,t2,t3,fpol,j1,den,dt2},
den=Sum[z[i](dL[momprop[[i]],momprop[[i]]]-masses[[i]]^2),{i,1,Length[momprop]}];
t1=-1/2D[den,#]/.{Derivative[0,1][dL]:>(#1&),Derivative[1,0][dL]:>(#2&)}&/@loopMom;
t2=-(D[t1,#])&/@loopMom;
t3=((-1/2D[den,#]/.{Derivative[0,1][dL]:>(lV[#1,\[Mu]]&),Derivative[1,0][dL]:>(lV[#2,\[Mu]]&)}&/@loopMom)/.Thread[loopMom->0]/.{lV[0,x_]:>0})//Simplify;
dt2=Factor[Det[t2]];
j1=dt2(den/.Thread[loopMom->0])/.{dL[0,X_]:>0};
fpol=Together[(-j1+Det[t2]t3 . (Inverse[t2]) . Transpose[{t3}])//Expand];
{dt2,fpol[[1]],t3,t2}
]
