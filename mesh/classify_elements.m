function classification = classify_elements(conn, phi)
% CLASSIFY_ELEMENTS Classifies elements into FULL_FLUID, FULL_SOLID, and CUT.
%   classification = classify_elements(conn, phi)
%   Fluid portion is phi < 0, Solid is phi > 0.
%   If all phi <= 0, element is FULL_FLUID.
%   If all phi > 0, element is FULL_SOLID.
%   Otherwise, element is CUT.

    types = {'tri3', 'quad4', 'tri6', 'quad9'};
    classification = struct();
    
    for t = 1:length(types)
        name = types{t};
        if ~isfield(conn, name) || isempty(conn.(name))
            continue;
        end
        
        elems = conn.(name);
        ne = size(elems, 1);
        
        is_fluid = false(ne, 1);
        is_solid = false(ne, 1);
        is_cut   = false(ne, 1);
        
        for e = 1:ne
            phi_e = phi(elems(e, :));
            
            if all(phi_e <= 0)
                is_fluid(e) = true;
            elseif all(phi_e > 0)
                is_solid(e) = true;
            else
                is_cut(e) = true;
            end
        end
        
        classification.(name).FULL_FLUID = find(is_fluid);
        classification.(name).FULL_SOLID = find(is_solid);
        classification.(name).CUT        = find(is_cut);
    end
end
