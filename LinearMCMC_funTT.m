function out1 = LinearMCMC_funTT(input1)
% Note: this program is built on Matlab (at least 2014 version) due to
% truncated distributed are used.
       
        ncohort      = input1.ncohort;
        oldthetaImat = input1.oldthetaImat;
        oldtheta     = input1.oldtheta;
        oldUmat      = input1.oldUmat;
        oldWmat      = input1.oldWmat;  
        oldtauE      = input1.oldtauE;
 
        oldInvSigma = input1.oldInvSigma;
        oldkappa     = input1.oldkappa;
        oldnu        = input1.oldnu;  
        oldlambda_kappa = input1.oldlambda_kappa;
        oldlambda_nu = input1.oldlambda_nu;

        sig1    = input1.sig1;
        sig2    = input1.sig2;
        sig3    = input1.sig3;
        p       = input1.p;
        tobs    = input1.tobs;
        nobs    = length(tobs);

        Yobsmat = input1.Yobsmat;
        D0Qbasismat = input1.D0Qbasismat;
        D1Qbasismat = input1.D1Qbasismat;
        quadwts = input1.quadwts;
        Phimat  = input1.Phimat;
        
        eta0    = input1.eta0;
        S01     = input1.S01;
        df      = input1.df;
        a       = input1.a;
        b       = input1.b;  
        c       = input1.c;
        d       = input1.d;
        
        InvOmega       = input1.InvOmega;
         
        countthetaI = input1.countthetaI;
        countkappa  = input1.countkappa;
        countnu     = input1.countnu;
        
        oldcparmat = input1.oldcparmat;
        XEstmat    = input1.XEstmat;
        options1   = input1.options1;
        
 
        %% Update theta_i
        
        myfitstr1.D0Qbasismat = D0Qbasismat;
        myfitstr1.D1Qbasismat = D1Qbasismat;
        myfitstr1.quadwts = quadwts;
        
        
    for ii=1:ncohort

        oldthetaI   = oldthetaImat(:,ii);   
        newthetaI   = oldthetaI+normrnd(0,sig1,p,1);

        oldU = oldUmat(ii);
        Yobs = Yobsmat(:,ii);

        c0  = oldthetaI(1);
        alpha = oldthetaI(2:p);
        myfitstr1.c0 = c0;
        myfitstr1.thetaI = alpha;
        oldcpar   = oldcparmat(:,ii);
        

        oldcpar = lsqnonlin(@LinearLSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        oldmu = Phimat*oldcpar;
        
        olddiff  = (Yobs-oldmu);  
        oldSSE   = -0.5*oldU*oldtauE*sum(olddiff.^2);
        oldprior = -0.5*oldWmat(ii)*(oldthetaI-oldtheta(1:p))'*oldInvSigma*(oldthetaI-oldtheta(1:p));
        den1     = oldSSE + oldprior;
        
        c0  = newthetaI(1);
        alpha = newthetaI(2:p);
        myfitstr1.c0 = c0;
        myfitstr1.thetaI = alpha;

        newcpar = lsqnonlin(@LinearLSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        newmu = Phimat*newcpar;

        newdiff  = (Yobs-newmu);  
        newSSE   = -0.5*oldU*oldtauE*sum(newdiff.^2);
        newprior = -0.5*oldWmat(ii)*(newthetaI-oldtheta(1:p))'*oldInvSigma*(newthetaI-oldtheta(1:p));
        num1     = newSSE + newprior;
       
        if log(rand(1))<=(num1-den1)
            oldthetaImat(:,ii) = newthetaI;
            countthetaI(ii)    = countthetaI(ii)+1;
            oldcparmat(:,ii)   = newcpar;
        else
            oldthetaImat(:,ii) = oldthetaI;

        end
    end

   
    %% Sample ui
    
    oldmu   = 0*Yobsmat;
    for i=1:ncohort
        C0 = oldthetaImat(1,i);
        alpha = oldthetaImat(2:p,i);
        oldmu(:,i) = Cfun(tobs,C0, alpha);
    end
    
    olddiff = Yobsmat-oldmu;
    resids2 = diag(olddiff'*olddiff);
    oldUmat = gamrnd(oldkappa/2+nobs/2,2./(oldkappa+oldtauE.*resids2),ncohort,1);
    
    %% Sample wi
    temp1 = oldthetaImat-oldtheta*ones(1,ncohort);
    temp2 = diag(temp1'*oldInvSigma*temp1);
    
    oldWmat = gamrnd(oldnu/2+p/2,2./(oldnu+temp2),ncohort,1);
 
    
    %% Sample kappa
    
    pd = makedist('normal',log(oldkappa),sig2);
    newnormpdf = truncate(pd,log(2.5),inf);
    newkappa = exp(random(newnormpdf,1));
    

    num1 = ncohort*(0.5*newkappa*log(0.5*newkappa)-gammaln(0.5*newkappa))+(0.5*newkappa-1)*sum(log(oldUmat))-0.5*newkappa*sum(oldUmat)-oldlambda_kappa*newkappa;
    den1 = ncohort*(0.5*oldkappa*log(0.5*oldkappa)-gammaln(0.5*oldkappa))+(0.5*oldkappa-1)*sum(log(oldUmat))-0.5*oldkappa*sum(oldUmat)-oldlambda_kappa*oldkappa;
    if log(rand(1))<=(num1-den1)
        oldkappa = newkappa;
        countkappa = countkappa+1;
    end

   %% Sample nu
   
     
    pd = makedist('normal',log(oldnu),sig3);
    newnormpdf = truncate(pd,log(2.5),inf);
    newnu = exp(random(newnormpdf,1));
    
    num1 = ncohort*(0.5*newnu*log(0.5*newnu)-gammaln(0.5*newnu))+(0.5*newnu-1)*sum(log(oldWmat))-0.5*newnu*sum(oldWmat)-oldlambda_nu*newnu;
    den1 = ncohort*(0.5*oldnu*log(0.5*oldnu)-gammaln(0.5*oldnu))+(0.5*oldnu-1)*sum(log(oldWmat))-0.5*oldnu*sum(oldWmat)-oldlambda_nu*oldnu;
    if log(rand(1))<=(num1-den1)
        oldnu = newnu;
        countnu = countnu+1;
    end

    %% Sample lambda_kappa and lambda_nu
        
    pd1 = makedist('gamma',2,1/oldkappa);
    newgampdf1 = truncate(pd1,c,d);
    newlambda_kappa = random(newgampdf1,1);
    pd2 = makedist('gamma',2,1/oldnu);
    newgampdf2 = truncate(pd2,c,d);
    newlambda_nu = random(newgampdf2,1);

    oldlambda_kappa = newlambda_kappa;
    oldlambda_nu = newlambda_nu;
       
    
    %% Sample theta;

    MuThetaI      = oldthetaImat*oldWmat;
    W             = sum(oldWmat)*oldInvSigma+InvOmega;
    postW         = inv(W);
    temp1         = oldInvSigma*MuThetaI+InvOmega*reshape(eta0,p,1);
    postmu        = W\temp1;
    oldtheta      = (mvnrnd(postmu, postW))';

    %% Sample inverse of SigmaThetaI;

    diffthetaI   = oldthetaImat-oldtheta*ones(1,ncohort);
    newOmega     = diffthetaI*diag(oldWmat)*diffthetaI'+S01;
    oldSigma = iwishrnd(newOmega, ncohort+df);
    oldInvSigma = inv(oldSigma);


    %% Sample tauE

    for i=1:ncohort
         c0 = oldthetaImat(1,i);
        alpha = oldthetaImat(2:p,i);
        myfitstr1.c0 = c0;
        myfitstr1.thetaI = alpha;
        oldcpar   = oldcparmat(:,ii);
        

        oldcpar = lsqnonlin(@LinearLSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        oldcparmat(:,ii) = oldcpar;
        XEstmat(:,ii) = Phimat*oldcpar;
    end
    
    diffmu = Yobsmat-XEstmat;
    SSE      = sum(oldUmat'.*sum(diffmu.^2));

    oldtauE = gamrnd(0.5*ncohort*nobs+a, 1/(0.5*SSE+1/b));
    
    %% Output.
    
    out1 = input1;
    
    out1.oldtheta     = oldtheta;
    out1.oldthetaImat = oldthetaImat;
    out1.oldmu        = oldmu;
    out1.oldUmat      = oldUmat;
    out1.oldtauE      = oldtauE;
    out1.oldkappa     = oldkappa;
    out1.oldWmat      = oldWmat;
    out1.oldnu        = oldnu;
    out1.oldlambda_kappa = oldlambda_kappa;
    out1.oldlambda_nu = oldlambda_nu;
    
    out1.oldInvSigma = oldInvSigma;
    
    out1.countthetaI  = countthetaI;
    out1.countkappa   = countkappa; 
    out1.countnu      = countnu; 
    
    out1.oldcparmat = oldcparmat;
    out1.XEstmat    = XEstmat;
           

 end

