function x = generateAnalyticalSolution(c, k, t)
    x0 = 1;
    v0 = 0;
    omega_n = sqrt(k);
    zeta = c / (2 * omega_n);

    if zeta < 1
        omega_d = omega_n * sqrt(1 - zeta^2);
        A = x0;
        B = (v0 + zeta * omega_n * x0) / omega_d;
        x = exp(-zeta * omega_n * t) .* (A * cos(omega_d * t) + B * sin(omega_d * t));
    else
        error('過減衰・臨界減衰は未対応');
    end
end
