function [A] = cordones(n)
%Juan Vera 2013
% generate the graph with high lovasz theta rank, as observed by juan and
% dobre

Jn = ones(n,n);
In = eye(n);
en = ones(n,1);
zn = zeros(n,1);
Zn = zeros(n,n);
A = logical([0  1  en'   zn'   zn'; 
     1  0  zn'   en'   zn'; 
	 en zn Zn    Jn-In In; 
	 zn en Jn-In Zn    In; 
	 zn zn In    In    Zn]); 
end