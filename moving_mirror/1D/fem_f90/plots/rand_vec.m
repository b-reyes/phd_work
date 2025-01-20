%creating a random vector, r_k where the values are between 1 and 5, and j is the length of r_k                                                                                      
function r_k = rand_vec(j,k)                                                                                                                                                                                                                                                                                                    

%Setting the seeds with respect to the system time and the rank 
rng((k+1)*cputime);

r_k = rand(j,1);  %r_k vector

%putting the numbers in the range [-.5,.5]                                                                                                                                              
for i = 1:j
   r_k(i,1) = r_k(i,1)*(.5 - -.5) + -.5;
end

end
