function phi = eval_levelset(nodes, cx, cy, R)
% EVAL_LEVELSET Evaluates the signed distance function phi(x,y)
%   phi = eval_levelset(nodes, cx, cy, R)
%   Fluid portion is phi < 0, Solid is phi > 0.
%   Inside cylinder: r < R -> R - r > 0 (solid)
%   Outside cylinder: r > R -> R - r < 0 (fluid)

    x = nodes(:, 1);
    y = nodes(:, 2);
    
    r = sqrt((x - cx).^2 + (y - cy).^2);
    phi = R - r;
end
