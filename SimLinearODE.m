% Simulation 1: parametric ODE


% Add path
%addpath('C:\Users\baisenl\Downloads\Code_ODE_HeavyTailed\fdaM')

odeoptions = odeset('RelTol',1e-7,'AbsTol',1e-7);
Lsqnonlinoptions = optimset('Display', 'iter','Tolfun',1e-7,'Largescale','off');

%% Generate simulated data.

ncohort = 50;
tobs = linspace(0,1,21)';
nobs = length(tobs);

tmin = min(tobs); tmax = max(tobs);
tEnd = tmax;

tfine    = linspace(tmin,tmax,101);
CSoluFine  = zeros(length(tfine), ncohort);

%% Generating the simulation data.

Nreps = 10;

sigma_epsilon = 0.4;

YobsmatArray_n50 = cell(Nreps,1);
TrueParaArray_n50 = cell(Nreps,1);
muSoluFArray_n50 = cell(Nreps,1);
mumatArray_n50 = cell(Nreps,1);

Yobsmat = zeros(nobs,ncohort);
mumat = zeros(nobs,ncohort);
muSoluF = zeros(length(tfine),ncohort);

for tt=1:Nreps
    display(num2str(tt));
    
    truea =  3;
    trueb =  10;

    truepar   = [truea; trueb];
    sigma_par = [0.5;1];
    
    thetaImat=truepar*ones(1,ncohort)+diag(sigma_par)*trnd(4,2,ncohort);

    y0 = 1;
    y0I = y0+0.2/sqrt(2)*trnd(4,ncohort,1);
    

    for i=1:ncohort
        mytheta = thetaImat(:,i);
        mystruct1.a = mytheta(1);
        mystruct1.b = mytheta(2);

        [tobs,myfit] = ode45(@LinearOde, tobs, y0I(i),odeoptions,mystruct1);   
       % myfit = Cfun(tobs,y0I(i),mytheta);
        mumat(:,i)=myfit;
        [tfine,Cfine] = ode45(@LinearOde, tfine, y0I(i),odeoptions,mystruct1);  
       % Cfine = Cfun(tfine,y0I(i),mytheta);
        muSoluF(:,i)=Cfine;
    end
    
    true_par = zeros(3,ncohort);
    true_par(1,:)=y0I';
    true_par(2:3,:)=thetaImat;
    
    TrueParaArray_n50{tt} = true_par;
    %TrueParaArray_n50{tt} =TruePar;
    
    muSoluFArray_n50{tt}=muSoluF;
    mumatArray_n50{tt}=mumat;

end

for tt=1:Nreps
    
mumat = mumatArray_n50{tt};
Yobsmat = mumat+trnd(4,nobs,ncohort);
YobsmatArray_n50{tt} = Yobsmat;

end

%% Preparing for MCMC algorithm.

tEnd = max(tobs);
knots   = linspace(0,tEnd,51)';
norder  = 4;
nbasis  = length(knots) + norder - 2;
basis   = create_bspline_basis([0,tEnd],nbasis,norder,knots);
J       = nbasis; 

% basis functions evaluated at tobs
Phimat   = eval_basis(tobs, basis);

% first derivative of basis functions evaluated at tobs
D1Phimat = eval_basis(tobs, basis, 1);
% second derivative of basis functions evaluated at tobs
D2Phimat = eval_basis(tobs, basis, 2);

% basis functions evaluated at tobs
Phimat_fine   = eval_basis(tfine, basis);

% first derivative of basis functions evaluated at tobs
D1Phimat_fine = eval_basis(tfine, basis, 1);
% second derivative of basis functions evaluated at tobs
D2Phimat_fine = eval_basis(tfine, basis, 2);


%% Calculate penalty term using Simpson's rule
h = 0.01;
quadpts = (0:h:tEnd)';
nquad = length(quadpts);
quadwts = ones(nquad,1);
quadwts(2:2:(nquad-1)) = 4;
quadwts(3:2:(nquad-2)) = 2;
quadwts = quadwts.*(h/3);

D0Qbasismat = eval_basis(quadpts, basis);
D1Qbasismat = eval_basis(quadpts, basis, 1);
D2Qbasismat = eval_basis(quadpts, basis, 2);
R1mat = D1Qbasismat'*(D1Qbasismat.*(quadwts*ones(1,nbasis)));
R2mat = D2Qbasismat'*(D2Qbasismat.*(quadwts*ones(1,nbasis)));

%% Set optimal criteria
options2 = optimset( ...
    'Algorithm', 'levenberg-marquardt', ...
    'LargeScale', 'off', ...
    'Display','off', ...
    'DerivativeCheck','off', ...
    'MaxIter', 100, ...
    'Jacobian','off',...
    'MaxFunEvals', 100000,...
    'TolFun', 1e-10, ...
    'TolX',   1e-10);

%% Start MCMC algorithm.

tfine = linspace(0, tmax, 101);
m_N = zeros(2,Nreps);
m_ST = zeros(2,Nreps);
Sim1_n50_thetaSam_TTArray = cell(Nreps,1);
Sim1_n50_thetaSam_NNArray = cell(Nreps,1);
thetaEst_NN=zeros(2,Nreps);
thetaEst_TT=zeros(2,Nreps);

Sim1_n50_XEst_TTArray = cell(Nreps,1);
Sim1_n50_XEst_NNArray = cell(Nreps,1);


for tt=1:Nreps
    display(num2str(tt));
    
true_par = TrueParaArray_n50{tt};
muSoluF = muSoluFArray_n50{tt};
mumat = mumatArray_n50{tt};
Yobsmat = YobsmatArray_n50{tt};

thetaIInt=true_par;

ITER = 2e2;

%% MCMC algorithm for Student-t/Student-t model.

p  = 3;

df  = p+1;
a   = 1; % shape parameter of gamma distribution.
b   = 100; % rate parameter of gamma distribution.
c   = 0.02;
d   = 0.5;

S01   = 0.01*eye(p); 
eta0     = [1; 3; 10]; 
InvOmega = 0.01*eye(p);
thetaInt = eta0+0.0000*normrnd(0, abs(eta0),p,1); 
truethetaI = thetaIInt;
InvSigma1Int = diag(exp(normrnd(-1,1,p,1)));

cparmat_TT = zeros(J,ncohort,ITER);
XEst_TT = zeros(nobs,ncohort,ITER);

tauESam_TT  = zeros(ITER,1); 
thetaSam_TT      = zeros(p,ITER);
thetaISam_TT     = zeros(p,ncohort,ITER);
InvSigmaSam_TT  = zeros(p,p,ITER);
USam_TT = zeros(ncohort,ITER);
WSam_TT = zeros(ncohort,ITER);
kappaSam_TT = zeros(ITER,1);
nuSam_TT = zeros(ITER,1);
lambdakappaSam_TT = zeros(ITER,1);
lambdanuSam_TT = zeros(ITER,1);

% Initial values;
tauESam_TT(1)   = 10;
kappaSam_TT(1) = 5;
nuSam_TT(1) = 5;
lambdakappaSam_TT(1)=0.1;
lambdanuSam_TT(1)=0.1;

thetaISam_TT(:,:,1)     = truethetaI;
thetaSam_TT(:,1)        = thetaInt;
InvSigmaSam_TT(:,:,1)   = InvSigma1Int;
USam_TT(:,1) = 3;
WSam_TT(:,1) = 3;

cparmat_TT(:,:,1) = (Phimat'*Phimat+0.001*R2mat)\(Phimat'*Yobsmat);
XEst_TT(:,:,1) = Phimat*cparmat_TT(:,:,1);

%  proposal;

sig1 = 0.15*[0.2;0.25; 0.25]; % For theta_i

sig2 = 0.5; % For kappa
sig3 = 0.5; % For nu

%%%%%%
% Gibbs sampler.

countthetaI = zeros(ncohort,1);
countkappa  = 0;
countnu     = 0;

oldthetaImat = thetaISam_TT(:,:,1);
oldtheta     = thetaSam_TT(:,1);
oldInvSigma  = InvSigmaSam_TT(:,:,1);
oldUmat      = USam_TT(:,1);
oldWmat      = WSam_TT(:,1);


oldtauE   = tauESam_TT(1);
oldkappa  = kappaSam_TT(1);
oldnu     = nuSam_TT(1);
oldlambda_kappa = lambdakappaSam_TT(1);
oldlambda_nu = lambdanuSam_TT(1);

oldcparmat = cparmat_TT(:,:,1);
XEstmat    = XEst_TT(:,:,1);

%%%%%%%%%%%
%% Start MCMC algorithm.

input1.ncohort = ncohort;

input1.oldthetaImat = oldthetaImat;
input1.oldtheta     = oldtheta;

input1.oldtauE   = oldtauE;
input1.oldkappa = oldkappa;
input1.oldnu  = oldnu;
input1.oldlambda_kappa = oldlambda_kappa;
input1.oldlambda_nu  = oldlambda_nu;

input1.oldUmat   = oldUmat;
input1.oldWmat   = oldWmat;

input1.sig1 = sig1;
input1.sig2 = sig2;
input1.sig3 = sig3;

input1.p    = p;

input1.tobs = tobs;
input1.Yobsmat = Yobsmat;
input1.D0Qbasismat = D0Qbasismat;
input1.D1Qbasismat = D1Qbasismat;
input1.quadwts = quadwts;
input1.Phimat  = Phimat;
input1.oldcparmat = oldcparmat;
input1.XEstmat = XEstmat;
input1.options1 = options2;

input1.eta0    = eta0;
input1.S01     = S01;
input1.df      = df;
input1.a       = a;
input1.b       = b;
input1.c       = c;
input1.d       = d;

input1.oldInvSigma = oldInvSigma;
input1.InvOmega    = InvOmega;


input1.countthetaI = countthetaI;
input1.countkappa  = countkappa;
input1.countnu     = countnu;

out1 = LinearMCMC_funTT(input1);
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

tic

for iter=1:ITER
    %display(num2str(iter))

    input1 = out1;
    
    out1 = LinearMCMC_funTT(input1);
    
    oldthetaImat = out1.oldthetaImat;
    oldtheta     = out1.oldtheta;
    oldInvSigma  = out1.oldInvSigma;
    oldtauE      = out1.oldtauE;

    oldUmat      = out1.oldUmat;
    oldWmat      = out1.oldWmat;
    oldkappa     = out1.oldkappa;
    oldnu        = out1.oldnu;
    oldlambda_kappa  = out1.oldlambda_kappa;
    oldlambda_nu     = out1.oldlambda_nu;
    
    oldcparmat   = out1.oldcparmat;
    XEstmat      = out1.XEstmat;
    
    
    countthetaI  = out1.countthetaI;
    countkappa   = out1.countkappa;
    countnu      = out1.countnu;
    
    thetaISam_TT(:,:,iter) = oldthetaImat;
    thetaSam_TT(:,iter)    = oldtheta;
    InvSigmaSam_TT(:,:,iter) = oldInvSigma;
    
    tauESam_TT(iter)       = oldtauE;
    USam_TT(:,iter)        = oldUmat;
    WSam_TT(:,iter)        = oldWmat;
    kappaSam_TT(iter)      = oldkappa;
    nuSam_TT(iter)         = oldnu;
    lambdakappaSam_TT(iter) = oldlambda_kappa;
    lambdanuSam_TT(iter)    = oldlambda_nu;

    cparmat_TT(:,:,iter) = oldcparmat;
    XEst_TT(:,:,iter)    = XEstmat;

end

toc


Sim1_n50_thetaSam_TTArray{tt}=thetaSam_TT;
Sim1_n50_XEst_TTArray{tt} = XEst_TT;

display(num2str(11111));

%% MCMC algorithm for Normal/Normal model.

tauESam_NN  = zeros(ITER,1); 
thetaSam_NN      = zeros(p,ITER);
thetaISam_NN     = zeros(p,ncohort,ITER);
InvSigmaSam_NN   = zeros(p,p,ITER);
cparmat_NN       = zeros(J,ncohort,ITER+1);
XEst_NN          = zeros(nobs,ncohort,ITER+1);


% Initial values;
tauESam_NN(1)   = 10;
thetaISam_NN(:,:,1)     = truethetaI;
thetaSam_NN(:,1)        = thetaInt;
InvSigmaSam_NN(:,:,1)   = InvSigma1Int;

cparmat_NN(:,:,1) = (Phimat'*Phimat+0.001*R2mat)\(Phimat'*Yobsmat);
XEst_NN(:,:,1)    = Phimat*cparmat_NN(:,:,1);

%  proposal;

%sig1 = 0.15*[0.2;0.25; 0.25; 0.5]; % For theta_i

%%%%%%
% Gibbs sampler.

countthetaI = zeros(ncohort,1);

oldthetaImat = thetaISam_NN(:,:,1);
oldtheta     = thetaSam_NN(:,1);
oldInvSigma  = InvSigmaSam_NN(:,:,1);

oldtauE   = tauESam_NN(1);

oldcparmat = cparmat_NN(:,:,1);
XEstmat    = XEst_NN(:,:,1);

%%%%%%%%%%%
%% Start MCMC algorithm.

input1.ncohort = ncohort;

input1.oldthetaImat = oldthetaImat;
input1.oldtheta     = oldtheta;

input1.oldtauE   = oldtauE;

input1.sig1 = sig1;

input1.p    = p;

input1.tobs = tobs;
input1.Yobsmat = Yobsmat;
input1.D0Qbasismat = D0Qbasismat;
input1.D1Qbasismat = D1Qbasismat;
input1.quadwts = quadwts;
input1.Phimat  = Phimat;
input1.oldcparmat = oldcparmat;
input1.XEstmat = XEstmat;
input1.options1 = options2;

input1.eta0    = eta0;
input1.S01     = S01;
input1.df      = df;
input1.a       = a;
input1.b       = b;

input1.oldInvSigma = oldInvSigma;
input1.InvOmega    = InvOmega;


input1.countthetaI = countthetaI;

out1 = LinearMCMC_funNN(input1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

tic

for iter=1:ITER
   % display(num2str(iter))

    input1 = out1;
    
    out1 = LinearMCMC_funNN(input1);
    
    oldthetaImat = out1.oldthetaImat;
    oldtheta     = out1.oldtheta;
    oldInvSigma  = out1.oldInvSigma;
    oldtauE      = out1.oldtauE;
    oldcparmat   = out1.oldcparmat;
    XEstmat      = out1.XEstmat;

    countthetaI  = out1.countthetaI;
    
    thetaISam_NN(:,:,iter) = oldthetaImat;
    thetaSam_NN(:,iter)    = oldtheta;
    InvSigmaSam_NN(:,:,iter) = oldInvSigma;
    
    tauESam_NN(iter)       = oldtauE;
    
    
    cparmat_NN(:,:,iter) = oldcparmat;
    XEst_NN(:,:,iter) = XEstmat;

end

toc


Sim1_n50_thetaSam_NNArray{tt}=thetaSam_NN;
Sim1_n50_XEst_NNArray{tt} = XEst_NN;

end


%% Analyzing.

truealpha = [3.0; 10.0];
index1 = (floor(ITER/2)+1):2:ITER;

for kk=1:Nreps
    temp1_NN = Sim1_n50_thetaSam_NNArray{kk};
    thetaEst_NN(:,kk) = mean(temp1_NN(2:p,index1),2);
    temp1_TT = Sim1_n50_thetaSam_TTArray{kk};
    thetaEst_TT(:,kk) = mean(temp1_TT(2:p,index1),2);
end

Mean_thetaEst_NN=mean(thetaEst_NN,2);
Mean_thetaEst_TT=mean(thetaEst_TT,2);
Std_thetaEst_NN = std(thetaEst_NN,0,2);
Std_thetaEst_TT = std(thetaEst_TT,0,2);
MADE_thetaEst_NN = mean(abs(thetaEst_NN-truealpha*ones(1,Nreps)),2);
MADE_thetaEst_TT = mean(abs(thetaEst_TT-truealpha*ones(1,Nreps)),2);
Results_n50_NN = [Mean_thetaEst_NN Std_thetaEst_NN MADE_thetaEst_NN];
Results_n50_TT = [Mean_thetaEst_TT Std_thetaEst_TT MADE_thetaEst_TT];

%% Finished.



        