D = data.frame( t(matrix(c(28,	99,
17,	78),nrow=2)))
require(exact2x2)

rS = rowSums(D)
cS = colSums(D)

res = uncondExact2x2(D[1,1],rS[1],D[2,1],rS[2])
summary(res)

print(help(uncondExact2x2))