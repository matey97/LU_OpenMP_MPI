function [ A, x, vPiv, err ] = LU_pivR( A, b, xObj );
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    % Initialization
    [m n] = size(A); dim = min([m n]);
    x = zeros (n,1); vPiv = [1:dim]';

    % This algorithm does not work for incompatible system
    if (m >= n),
        % Gaussian Elimination (row pivoting)
        for k = 1:dim,
            [piv,ipiv] = max(abs(A(k,k:n)));
            ipiv = ipiv+k-1;
            piv = A(k,ipiv); 
            if (ipiv ~= k),
                vtmp = A(:,k);
                A(:,k) = A(:,ipiv);
                A(:,ipiv) = vtmp;
                vtmp = vPiv(k);
                vPiv(k) = vPiv(ipiv);
                vPiv(ipiv) = vtmp;
            end
            A(k,k:n) = A(k,k:n)/piv;
            b(k)     = b(k)/piv;        
            A(k+1:m,k+1:n) = A(k+1:m,k+1:n) - A(k+1:m,k) * A(k,k+1:n);
            b(k+1:m) = b(k+1:m) - A(k+1:m,k) * b(k);
        end

        % Backward substitution
        for k = dim:-1:1,
            for i=1:k-1,
                b(i) = b(i) - b(k) * A(i,k);
            end;
        end

        % Remove permutation
        for i=1:n, x(vPiv(i)) = b(i); end;

        % Error computation
        err = norm(x-xObj);
    else,
        disp ('This algorithm does not work for incompatible systems')
        err = 0;
    end;
end

