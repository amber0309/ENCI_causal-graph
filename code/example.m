%{ 

an illustrative example to show the usage of ENCI_graph()

Shoubo (shoubo.sub AT gmail.com)
05/04/2018
%}

[X, sklt] = gen_toy(1000, 'mtp', 40, 'gmm')

[B, prc, rcl] = ENCI_graph(X, sklt)