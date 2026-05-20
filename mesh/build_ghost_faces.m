function ghost_info = build_ghost_faces(faces, classification)
% BUILD_GHOST_FACES Identifies edges shared between CUT and FULL_FLUID elements.
%   ghost_info = build_ghost_faces(faces, classification)
%   Returns a struct with edges, elem_L, elem_R, type_L, type_R

    nFaces = size(faces.edges, 1);
    is_ghost = false(nFaces, 1);
    
    for i = 1:nFaces
        type_L = faces.type_L{i};
        type_R = faces.type_R{i};
        local_L = faces.local_L(i);
        local_R = faces.local_R(i);
        
        if local_R == 0
            continue; % Boundary face, not internal
        end
        
        % Check Element L
        L_is_cut = ismember(local_L, classification.(type_L).CUT);
        L_is_fluid = ismember(local_L, classification.(type_L).FULL_FLUID);
        
        % Check Element R
        R_is_cut = ismember(local_R, classification.(type_R).CUT);
        R_is_fluid = ismember(local_R, classification.(type_R).FULL_FLUID);
        
        % An edge is a ghost penalty edge if at least one element is CUT 
        % and both elements belong to the active domain (CUT or FULL_FLUID).
        if (L_is_cut || R_is_cut) && (L_is_cut || L_is_fluid) && (R_is_cut || R_is_fluid)
            is_ghost(i) = true;
        end
    end
    
    ghost_info.edges = faces.edges(is_ghost, :);
    ghost_info.elem_L = faces.local_L(is_ghost);
    ghost_info.elem_R = faces.local_R(is_ghost);
    ghost_info.type_L = faces.type_L(is_ghost);
    ghost_info.type_R = faces.type_R(is_ghost);
end
