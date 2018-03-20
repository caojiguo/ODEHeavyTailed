function dy = LinearOde(t,y,input)
   
    a = input.a;
    b = input.b;
    
    dy  = -a*y+b;
    
end