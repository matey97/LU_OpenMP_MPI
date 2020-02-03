function [ A, x, vPiv, err] = LU_pivC( A, b, xObj )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    % Initialization
    [m n] = size(A); dim = min([m n]);
    x = zeros (n,1); vPiv = [1:dim]'; 
    
    % This algorithm does not work for incompatible system
    if (m >= n),

        % Gaussian Elimination (column pivoting)
        for k = 1:dim,
            [piv,ipiv] = max(abs(A(k:m,k)));
            ipiv = ipiv+k-1;
            piv = A(ipiv,k); vPiv(k) = ipiv;
            if (ipiv ~= k),
                vtmp = A(k,:);
                A(k,:) = A(ipiv,:);
                A(ipiv,:) = vtmp;
                vtmp = b(k);
                b(k) = b(ipiv);
                b(ipiv) = vtmp;
            end
            A(k+1:m,k) = A(k+1:m,k)/piv;
            A(k+1:m,k+1:n) = A(k+1:m,k+1:n) - A(k+1:m,k) * A(k,k+1:n);
            b(k+1:m) = b(k+1:m) - A(k+1:m,k) * b(k);
        end

        % Backward substitution
        for k = dim:-1:1,
            for j=k+1:dim,
                b(k) = b(k) - b(j) * A(k,j);
            end;
            b(k) = b(k) / A(k,k);
        end
         
        % Obtain the solution
        x = b(1:dim);
        
        % Error computation
        err = norm(b(1:dim)-xObj(1:dim));
        
    else
        disp ('This algorithm does not work for incompatible systems')
        err = 0;
    end;
end
