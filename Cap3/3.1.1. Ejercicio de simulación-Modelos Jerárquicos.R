###################################################################
#Subsecci?n 3.1.1- Ejercicio de simulaci?n-Modelos jer?rquicos.

#Fecha de ?ltima modificaci?n: 10-oct-22.

#Tesis: Modelaci?n de la relaci?n entre el cerebro y el comportamiento mediante campos Gaussianos.

#Autor: James J. Palestro a, Giwon Bahg a, Per B. Sederberg c, Zhong-Lin Lu a, Mark Steyvers b,Brandon M. Turner

#Modificado por: Daniela Mart?nez Aguirre

#Descripci?n: C?digo extraido y modificado del art?culo: https://doi.org/10.1016/j.jmp.2018.03.003

#Figuras: 3.2, 3.3, 3.5 y 3.6

###################################################################


#Instalaci?n de librer?a Jags y necesarias para implementar
install.packages( "rjags" )
require( "rjags" )
#install.packages( "gdata" )
#install.packages( "GoFKernel" )
library(gdata)
library(GoFKernel)
library(mvtnorm)
library(rjags)
library(ssize)
#install.packages( "tseries" )
library(tseries)


#LImpiamos ?rea de trabajo
rm(list = ls())
#Establecemos una semilla
set.seed(1)

#Definimos la funci?n logit y su inversa
# need both logit and logit^{-1} functions
logit=function(x)log(x/(1-x))
invlogit=function(x){1/(1+exp(-x))}
x<-seq(-4,4,length.out = 100)

#Gr?fica de la funci?n logit inversa
# Lista (expression vector) con los textos de la leyenda
l <- expression(paste(theta,"=-2"), paste(theta,"= 0"), paste(theta,"= 2"))

#Figura 3.3
#png("logit_inversa.png" ,width = 300, height = 200, units='mm', res = 300)
plot(x, invlogit(x), xlab=expression(theta), ylab="Probabilidad de recordar",lwd = 3, cex.lab=1.5,col="black", type="l")
lines(seq(-5,0,length.out = 100), rep(invlogit(0),100),col="blue", lty=2, lwd = 3)
lines(rep(0,100),seq(0,invlogit(0),length.out = 100),col="blue", lty=2, lwd=3)
lines(seq(-5,-2,length.out = 100), rep(invlogit(-2),100),col="green", lty=2, lwd = 3)
lines(rep(-2,100),seq(0,invlogit(-2),length.out = 100),col="green", lty=2, lwd = 3)
lines(seq(-5,2,length.out = 100), rep(invlogit(2),100),col="red", lty=2,lwd = 3)
lines(rep(2,100),seq(0,invlogit(2),length.out = 100),col="red", lty=2,lwd = 3)
legend("topright",          
       legend = l,        
       lty = c(2, 2,2), 
       bty = "n",          
       col = c("green", "blue","red"),
       inset = .05,         
       y.intersp = 1, cex=2) 
grid()
#points(X,y , col="red")
#dev.off()


###Par?metros del modelo

n <- 100 # n?mero de im?genes n/2 repetidas y n/2 nuevas
p<-30 #numero de pacientes
#Orden real de las im?genes
orden<-c(rep(0,n/2), rep(1,n/2))
orden<-sample(orden)

#Par?metros del modelo conjunto, phi y sigma
#Establecer par?emtros delta, el modelo lineal con sus sd para cada una de las
#regiones de activaci?n
sig1 <- .5 # std. dev. of single -trial BOLD responses , ROI 1
sig2 <- 1 # std. dev. of item memory strength (logit scale)
rho <- .8 # cor b/n brain activation and memory strength
rho2 <- 0 # correlaci?n entre los dos par?metros de activaci?n
rho3 <- 0 # correlaci?n entre los dos par?metros memoria
rho4<- 0 #Correlaci?n entre el par?metro de memorio 1 con de de activaci?n 0 y viceversa

#Creamos la matriz de covarianza del modelo neuronal para los par?metros theta, deltha
sigma <- matrix(c(sig1^2, sig1*sig1*rho2,sig1*sig2*rho,sig1*sig2*rho4,
                  sig1*sig1*rho2, sig1^2,sig1*sig2*rho4,sig1*sig2*rho,
                  sig2*sig1*rho, sig2*sig1*rho4,sig2^2, sig2*sig2*rho3,
                  sig2*sig1*rho4, sig2*sig1*rho, sig2*sig2*rho3, sig2^2 ), # element [2,2]
                4,4,byrow=TRUE)



#Es la media de los datos para 
#elta1 de imagen nueva, 
#delta2 de imagen repetida
#theta1 de imagen nueva
#theta2 de imagen repetida
phi <- c(1.5,2,-1,2 )


###Simulaci?n de datos


#Se simula los par?metros delta theta de los pacientes.
#Simulate single -trial delta matrix
DeltaTheta <- rmvnorm(p,phi,sigma)

#Ahora generamos los datos del modelo neuronal y del comportamiento
#Genera los tiempos
#Tiempos
ts <- seq(0,4,1) # scan times
sig <- .5 # the std. dev. of BOLD responses

#Declarar tablas donde guardaremos los datos
N <- matrix(NA,n*p,length(ts))
B <- numeric(n*p)

#for para in paciente
for (j in 1:p){
  for(i in 1:n){
    #Creamos una normal con las delthatheta que representan deltas agregando un error
    N[i+(j-1)*n,]=rnorm(length(ts),DeltaTheta[j,orden[i]+1]*ts,sig)
    
    B[i+(j-1)*n]=rbinom(1,1,invlogit(DeltaTheta[j,orden[i]+3]))
    
  }
}

#Figura 3.2
#png("Datos_neuronales.png" ,width = 300, height = 200, units='mm', res = 300)
plot(N[1,],  xlab="T", ylab="grado de activaci?n (BOLD)", col="red", type="b", cex.lab=1.5, lwd=2)
lines(N[2,],col="blue", type="b", lwd=2)
lines(N[3,], col="green", type="b", lwd=2)
grid()
#points(X,y , col="red")
#dev.off()

###Definici?n del modelo jer?rquico y algorimo MCMC

#Creamos variables para inicializar en  JAGS
dat = list("n"=n, "B"=B,
           "N"=N,
           "ts"=ts,
           "Nt"=length(ts),
           "sig"=sig,
           "I0"=diag(4),
           "n0"=4,
           "phi0"=rep(0,4),
           "s0"=diag(4), "orden"=orden, "p"=p)
Nt=5

#Define el modelo jer?rquico
jags <- jags.model("modelo2.txt",
                   data = dat,
                   n.chains = 1,
                   n.adapt = 10000)


#Continua adaptando
while (FALSE){
  adapt(jags , 1000, end.adaptation=FALSE)
}
update(jags , 2000)

### Obtenci?n de muestras

#N?mero de muestras
m=1000000

#Guardamos las muestras
out=jags.samples(jags ,c("phi", "Sigma", "DeltaTheta"),m)

#Nos quedaremos con cada adel=100 iteraciones para evitar autocorrelaci?n
adel=100
z=seq(1,m,adel)
out$DeltaTheta=out$DeltaTheta[,,z,]
out$Sigma=out$Sigma[,,z,]
out$phi=out$phi[,z,]


#Calculate la media de la posterior
pms=apply(out$DeltaTheta ,c(1,2),median)
sd=apply(out$Sigma ,c(1,2),median)
pi=apply(out$phi ,1,median)

#Coeficiente de correlaci?n
ro1_es=c(sd[1,3])/(sqrt(sd[1,1])*sqrt(sd[3,3]))
ro2_es=c(sd[2,4])/(sqrt(sd[4,4])*sqrt(sd[2,2]))


#Delta 0 primera columna
#Delta 1 segunda columna
#theta 0 tercera columna
#Theta 1 Cuarta columna
delta0=pms[,1]
delta1=pms[,2]
theta0=pms[,3]
theta1=pms[,4]

###An?lisis y resultados

#Creamos una secuencia
x=seq(-3,5,length.out = 1000)

#Figura 3.6
#Graficas de thetas y deltas
#png("delta0.png" ,width = 300, height = 200, units='mm', res = 300)
plot(DeltaTheta[,1], delta0, lwd=5,col="purple", xlab=expression(paste(" Valor real ",delta[0])) , ylab=expression(paste(" Valor estimado ",delta[0])), cex.lab=1.2)
lines(x,x, type="l", col="black" )
grid()
#points(X,y , col="red")
#dev.off()

#png("delta1.png" ,width = 300, height = 200, units='mm', res = 300)
plot(DeltaTheta[,2], delta1, lwd=5,col="purple", xlab=expression(paste(" Valor real ",delta[1])) , ylab=expression(paste(" Valor estimado ",delta[1])), cex.lab=1.2)
lines(x,x, type="l", col="black" )
grid()
#points(X,y , col="red")
#dev.off()

#png("theta0.png" ,width = 300, height = 200, units='mm', res = 300)
plot(DeltaTheta[,3], theta0, lwd=5,col="purple", xlab=expression(paste(" Valor real ",theta[0])) , ylab=expression(paste(" Valor estimado ",theta[0])), cex.lab=1.2)
lines(x,x, type="l", col="black" )
grid()
#points(X,y , col="red")
#dev.off()

#Aplicando logit inversa
#png("theta0_prob.png" ,width = 300, height = 200, units='mm', res = 300)
plot(invlogit(DeltaTheta[,3]),invlogit( theta0), lwd=5,col="purple", xlab=expression(paste(" Valor real logit inv(",theta[0],")")) , ylab=expression(paste(" Valor estimado logit inv( ",theta[0],")")), cex.lab=1.2)
lines(x,x, type="l", col="black" )
grid()
#points(X,y , col="red")
#dev.off()

#png("theta1.png" ,width = 300, height = 200, units='mm', res = 300)
plot(DeltaTheta[,4], theta1, lwd=5,col="purple", xlab=expression(paste(" Valor real ",theta[1])) , ylab=expression(paste(" Valor estimado ",theta[1])), cex.lab=1.2)
lines(x,x, type="l", col="black" )
grid()
#points(X,y , col="red")
#dev.off()

#png("theta1_prob.png" ,width = 300, height = 200, units='mm', res = 300)
plot(invlogit(DeltaTheta[,4]),invlogit( theta1), lwd=5,col="purple", xlab=expression(paste(" Valor real logit inv(",theta[1],")")) , ylab=expression(paste(" Valor estimado logit inv( ",theta[1],")")), cex.lab=1.2)
lines(x,x, type="l", col="black" )
grid()
#points(X,y , col="red")
#dev.off()


#Extraccipn de los par?metros sd de delta
e_sd_delta0=sqrt(c(out$Sigma[1,1,]))
e_sd_delta1=sqrt(c(out$Sigma[2,2,]))

#calculate autocorrelations
acf(e_sd_delta0, pl=FALSE)
acf(e_sd_delta1, pl=FALSE)

#Figura 3.5 c
png("sd_delta.png" ,width = 300, height = 200, units='mm', res = 300)
boxplot(cbind(e_sd_delta0,e_sd_delta1) , col="deepskyblue", cex.lab=2, axes = FALSE)
abline(h=0.5, col="red", lty=2, lwd=3)
axis(side=1, at=1:2, labels=c(expression(sigma[delta[0]]) ,expression(sigma[delta[1]])), cex.axis=2)
axis(2)
box() 
#grid()
dev.off()

#Extracci?n de los par?metros sd de delta
e_sd_theta0=sqrt(c(out$Sigma[3,3,]))
e_sd_theta1=sqrt(c(out$Sigma[4,4,]))
acf(e_sd_theta0, pl=FALSE)
acf(e_sd_theta1, pl=FALSE)

#Figura 3.5 d
#png("sd_theta.png" ,width = 300, height = 200, units='mm', res = 300)
boxplot(cbind(e_sd_theta0,e_sd_theta1) , col="deepskyblue", cex.lab=2, axes = FALSE)
abline(h=1, col="red", lty=2, lwd=3)
axis(side=1, at=1:2, labels=c(expression(sigma[theta[0]]) ,expression(sigma[theta[1]])), cex.axis=2)
axis(2)
box() 
#grid()
#dev.off()

mean_delta0=c(out$phi[1,])
mean_delta1=c(out$phi[2,])
mean_tetha0=c(out$phi[3,])
mean_tetha1=c(out$phi[4,])

#Calculo de autocorrelacion
acf(mean_delta0, pl=FALSE)
acf(mean_delta1, pl=FALSE)
acf(mean_tetha0, pl=FALSE)
acf(mean_tetha1, pl=FALSE)

#Figura 3.5 a
#png("phi.png" ,width = 300, height = 200, units='mm', res = 300)
boxplot(cbind(mean_delta0,mean_delta1,mean_tetha0, mean_tetha1 ) , col="deepskyblue", cex.lab=2, axes = FALSE)
#abline(h=1, col="red", lty=3, lwd=2)
axis(side=1, at=1:4, labels=c(expression(phi[1]) ,expression(phi[2]), expression(phi[3]), expression(phi[4])), cex.axis=2)
lines(seq(.5,1.5,length.out = 100), rep(1.5,100),col="red", lty=2, lwd = 3)
lines(seq(1.5,2.5,length.out = 100), rep(2,100),col="red", lty=2, lwd = 3)
lines(seq(2.5,3.5,length.out = 100), rep(-1,100),col="red", lty=2, lwd = 3)
lines(seq(3.5,4.5,length.out = 100), rep(2,100),col="red", lty=2, lwd = 3)
axis(2)
box() 
#grid()
#dev.off()

#Figura 3.5 a
#png("phi1.png" ,width = 300, height = 200, units='mm', res = 300)
boxplot(cbind(mean_tetha0, mean_tetha1 ) , col="deepskyblue", cex.lab=2, axes = FALSE)
#abline(h=1, col="red", lty=3, lwd=2)
axis(side=1, at=1:2, labels=c(expression(phi[3]) ,expression(phi[4])), cex.axis=2)
axis(2)
box() 
#grid()
#dev.off()

#png("phi2.png" ,width = 300, height = 200, units='mm', res = 300)
boxplot(cbind(mean_delta0,mean_delta1 ) , col="deepskyblue", cex.lab=2, axes = FALSE)
#abline(h=1, col="red", lty=3, lwd=2)
axis(side=1, at=1:2, labels=c(expression(phi[1]) ,expression(phi[2])), cex.axis=2)
axis(2)
box() 
#grid()
#dev.off()


#Extracci?n de los par?metros rho

ro1=c(out$Sigma[1,3,])/(e_sd_delta0*e_sd_theta0)
ro2=c(out$Sigma[2,4,])/(e_sd_delta1*e_sd_theta1)
acf(ro1, pl=FALSE)
acf(ro2, pl=FALSE)

#Figura 3.5 b
#png("rho.png" ,width = 300, height = 200, units='mm', res = 300)
boxplot(cbind( ro1,ro2) , col="deepskyblue", cex.lab=2, axes = FALSE)
abline(h=0.8, col="red", lty=2, lwd=3)
axis(side=1, at=1:2, labels=c(expression(rho[1]) ,expression(rho[2])), cex.axis=2)
axis(2)
box() 
#grid()
#dev.off()


#Ver la cadena
plot(e_sd_theta0[1:100])
plot(e_sd_theta1[1:100])
plot(e_sd_delta0[1:100])
plot(e_sd_delta1[1:100])

plot(mean_tetha0[1:100])
plot(mean_tetha1[1:100])
plot(mean_delta0[1:100])
plot(mean_delta1[1:100])

plot(ro1[1:100])
plot(ro2[1:100])
