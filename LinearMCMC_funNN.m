function out1 = LinearMCMC_funNN(input1)

       
        ncohort      = input1.ncohort;
        oldthetaImat = input1.oldthetaImat;
        oldtheta     = input1.oldtheta;
        oldtauE      = input1.oldtauE;
 
        oldInvSigma = input1.oldInvSigma;

        sig1    = input1.sig1;

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
        
        InvOmega    = input1.InvOmega;
         
        countthetaI = input1.countthetaI;
        
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

        Yobs = Yobsmat(:,ii);

        c0  = oldthetaI(1);
        alpha = oldthetaI(2:p);
        myfitstr1.c0 = c0;
        myfitstr1.thetaI = alpha;
        oldcpar   = oldcparmat(:,ii);
        

        oldcpar = lsqnonlin(@LinearLSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        oldmu = Phimat*oldcpar;

        olddiff  = (Yobs-oldmu);  
        oldSSE   = -0.5*oldtauE*sum(olddiff.^2);
        oldprior = -0.5*(oldthetaI-oldtheta(1:p))'*oldInvSigma*(oldthetaI-oldtheta(1:p));
        den1     = oldSSE + oldprior;


        c0  = newthetaI(1);
        alpha = newthetaI(2:p);
        myfitstr1.c0 = c0;
        myfitstr1.thetaI = alpha;

        newcpar = lsqnonlin(@LinearLSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        newmu = Phimat*newcpar;

        newdiff  = (Yobs-newmu);  
        newSSE   = -0.5*oldtauE*sum(newdiff.^2);
        newprior = -0.5*(newthetaI-oldtheta(1:p))'*oldInvSigma*(newthetaI-oldtheta(1:p));
        num1     = newSSE + newprior;
       
        if log(rand(1))<=(num1-den1)
            oldthetaImat(:,ii) = newthetaI;
            countthetaI(ii)    = countthetaI(ii)+1;
            oldcparmat(:,ii)   = newcpar;
        else
            oldthetaImat(:,ii) = oldthetaI;

        end
    end

    
    %% Sample theta;

    MuThetaI      = oldthetaImat*ones(ncohort,1);
    W             = ncohort*oldInvSigma+InvOmega;
    postW         = inv(W);
    temp1         = oldInvSigma*MuThetaI+InvOmega*reshape(eta0,p,1);
    postmu        = W\temp1;
    oldtheta      = (mvnrnd(postmu, postW))';

    %% Sample inverse of SigmaThetaI;

    diffthetaI   = oldthetaImat-oldtheta*ones(1,ncohort);
    newOmega     = diffthetaI*diffthetaI'+S01;
    oldSigma = iwishrnd(newOmega, ncohort+df);
    oldInvSigma = inv(oldSigma);


    %% Sample tauE

    
    for i=1:ncohort
        c0 = oldthetaImat(1,i);
        alpha = oldthetaImat(2:p,i);
        myfitstr1.c0 = c0;
        myfitstr1.thetaI = alpha;
        oldcpar   = oldcparmat(:,i);
        

        oldcpar = lsqnonlin(@LinearLSQNONLIN_fun, oldcpar,[],[],options1,myfitstr1);
        oldcparmat(:,i) = oldcpar;
        XEstmat(:,i) = Phimat*oldcpar;
    end
    
    diffmu = Yobsmat-XEstmat;
    SSE      = sum(sum(diffmu.^2));

    oldtauE = gamrnd(0.5*ncohort*nobs+a, 1/(0.5*SSE+1/b));
    
    %% Output.
    
    out1 = input1;
    
    out1.oldtheta     = oldtheta;
    out1.oldthetaImat = oldthetaImat;
    out1.oldmu        = oldmu;
    out1.oldtauE      = oldtauE;
    
    out1.oldInvSigma = oldInvSigma;
    
    out1.countthetaI  = countthetaI;
    
    out1.oldcparmat = oldcparmat;
    out1.XEstmat    = XEstmat;

 end

