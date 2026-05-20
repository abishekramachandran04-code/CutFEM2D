function [I, J_idx, V, R_global] = navierstokes_jacobian(mesh, conn, edof, dof, Re, dt, U_k, U_n, f_source)
% NAVIERSTOKES_JACOBIAN  GPU-vectorised assembly of the exact Tangent
% Stiffness Matrix (Jacobian) and Non-Linear Residual for Newton-Raphson.

    total_dof = dof.ndof;
    ndpn = 3;
    R_cpu = zeros(total_dof, 1);
    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
    I_all = []; J_all = []; V_all = [];

    U_k_g = gpuArray(U_k(:));
    U_n_g = gpuArray(U_n(:));
    nodes_g = gpuArray(mesh.nodes(:,1:2));

    for t = 1:length(elem_types)
        type = elem_types{t};
        if ~isfield(conn, type) || isempty(conn.(type)), continue; end

        elems = conn.(type); edofs = edof.(type);
        ne = size(elems,1); nen = size(elems,2); ndof_e = nen*ndpn;
        [xi,eta,wt] = gauss_quadrature(type); nGP = length(wt);
        is_quad = (nen==6||nen==9);
        elems_g = gpuArray(elems); edofs_g = gpuArray(edofs);

        idx_f = reshape(elems_g',[],1); crd_f = nodes_g(idx_f,:);
        coords = permute(reshape(crd_f,nen,ne,2),[1 3 2]);
        ed_f = reshape(edofs_g',[],1);
        Uk_l = reshape(U_k_g(ed_f),ndof_e,ne); Un_l = reshape(U_n_g(ed_f),ndof_e,ne);
        uk=Uk_l(1:3:end,:); vk=Uk_l(2:3:end,:); un=Un_l(1:3:end,:); vn=Un_l(2:3:end,:);

        Ae=zeros(1,1,ne,'gpuArray');
        for q=1:nGP, [~,dNq]=shape_funcs(xi(q),eta(q),type); Jq=pagemtimes(gpuArray(dNq),coords); Ae=Ae+(Jq(1,1,:).*Jq(2,2,:)-Jq(1,2,:).*Jq(2,1,:))*wt(q); end
        if contains(type,'tri'),nc=3;else,nc=4;end
        corn=coords(1:nc,:,:); nxt=[corn(2:end,:,:);corn(1,:,:)];
        P_e=sum(sqrt(sum((nxt-corn).^2,2)),1); h_e=4*Ae./P_e;
        um=reshape(mean(uk,1),1,1,ne); vm=reshape(mean(vk,1),1,1,ne); umag=sqrt(um.^2+vm.^2);
        tau_s=1./sqrt((2/dt)^2+(2*umag./h_e).^2+(4./(Re*h_e.^2)).^2);

        Ke_pic=zeros(ndof_e,ndof_e,ne,'gpuArray'); Ke_tan=zeros(ndof_e,ndof_e,ne,'gpuArray');
        Fe_pic=zeros(ndof_e,1,ne,'gpuArray');
        iu=1:3:ndof_e; iv=2:3:ndof_e; ip=3:3:ndof_e;

        for q=1:nGP
            [N,dNxi]=shape_funcs(xi(q),eta(q),type); N=gpuArray(N(:)); dNxi=gpuArray(dNxi);
            Jm=pagemtimes(dNxi,coords); dJ=Jm(1,1,:).*Jm(2,2,:)-Jm(1,2,:).*Jm(2,1,:);
            iJ=zeros(2,2,ne,'gpuArray');
            iJ(1,1,:)=Jm(2,2,:)./dJ; iJ(1,2,:)=-Jm(1,2,:)./dJ;
            iJ(2,1,:)=-Jm(2,1,:)./dJ; iJ(2,2,:)=Jm(1,1,:)./dJ;
            dNx=pagemtimes(iJ,repmat(dNxi,1,1,ne)); dV=dJ*wt(q);

            if is_quad
                d2=gpuArray(shape_funcs_2nd(xi(q),eta(q),type));
                s11=iJ(1,1,:);s12=iJ(1,2,:);s21=iJ(2,1,:);s22=iJ(2,2,:);
                lap=(s11.^2+s21.^2).*d2(1,:)+2*(s11.*s12+s21.*s22).*d2(3,:)+(s12.^2+s22.^2).*d2(2,:);
            else, lap=zeros(1,nen,ne,'gpuArray'); end

            xg=sum(N.*coords(:,1,:),1); yg=sum(N.*coords(:,2,:),1);
            fx=f_source{1}(xg,yg); fy=f_source{2}(xg,yg);
            ugk=reshape(N'*uk,1,1,ne); vgk=reshape(N'*vk,1,1,ne);
            ugn=reshape(N'*un,1,1,ne); vgn=reshape(N'*vn,1,1,ne);

            Ladv=ugk.*dNx(1,:,:)+vgk.*dNx(2,:,:);
            Lcol=permute(Ladv,[2 1 3]); Nt=reshape(N',1,nen);
            dNxC=permute(dNx(1,:,:),[2 1 3]); dNyC=permute(dNx(2,:,:),[2 1 3]);

            M0=N*Nt; Msupg=tau_s.*pagemtimes(Lcol,Nt);
            Kvisc=(1/Re)*pagemtimes(permute(dNx,[2 1 3]),dNx);
            Kadv=pagemtimes(N,Ladv); Ksadv=tau_s.*pagemtimes(Lcol,Ladv);
            Ksdif=tau_s.*(-(1/Re)).*pagemtimes(Lcol,lap);
            Kuu=(M0+Msupg+dt*(Kvisc+Kadv+Ksadv+Ksdif)).*dV;
            Kup=dt*(-1).*pagemtimes(dNxC,Nt).*dV; Kvp=dt*(-1).*pagemtimes(dNyC,Nt).*dV;
            Mpx=tau_s.*pagemtimes(dNxC,Nt); Mpy=tau_s.*pagemtimes(dNyC,Nt);
            Dx=pagemtimes(N,dNx(1,:,:)); Dy=pagemtimes(N,dNx(2,:,:));
            Dpx=tau_s.*pagemtimes(dNxC,Ladv); Dpy=tau_s.*pagemtimes(dNyC,Ladv);
            Pvx=-(tau_s/Re).*pagemtimes(dNxC,lap); Pvy=-(tau_s/Re).*pagemtimes(dNyC,lap);
            Kpu=(Mpx+dt*(Dx+Dpx+Pvx)).*dV; Kpv=(Mpy+dt*(Dy+Dpy+Pvy)).*dV;
            Kpp=(dt.*tau_s).*pagemtimes(permute(dNx,[2 1 3]),dNx).*dV;

            Ke_pic(iu,iu,:)=Ke_pic(iu,iu,:)+Kuu; Ke_pic(iv,iv,:)=Ke_pic(iv,iv,:)+Kuu;
            Ke_pic(iu,ip,:)=Ke_pic(iu,ip,:)+Kup; Ke_pic(iv,ip,:)=Ke_pic(iv,ip,:)+Kvp;
            Ke_pic(ip,iu,:)=Ke_pic(ip,iu,:)+Kpu; Ke_pic(ip,iv,:)=Ke_pic(ip,iv,:)+Kpv;
            Ke_pic(ip,ip,:)=Ke_pic(ip,ip,:)+Kpp;

            NpL=N+tau_s.*Lcol;
            Fe_pic(iu,1,:)=Fe_pic(iu,1,:)+NpL.*(dt*fx+ugn).*dV;
            Fe_pic(iv,1,:)=Fe_pic(iv,1,:)+NpL.*(dt*fy+vgn).*dV;
            Fe_pic(ip,1,:)=Fe_pic(ip,1,:)+tau_s.*(dt*(dNxC.*fx+dNyC.*fy)+(dNxC.*ugn+dNyC.*vgn)).*dV;

            dudx_k=reshape(sum(squeeze(dNx(1,:,:)).*uk,1),1,1,ne);
            dudy_k=reshape(sum(squeeze(dNx(2,:,:)).*uk,1),1,1,ne);
            dvdx_k=reshape(sum(squeeze(dNx(1,:,:)).*vk,1),1,1,ne);
            dvdy_k=reshape(sum(squeeze(dNx(2,:,:)).*vk,1),1,1,ne);

            NNt=N*Nt; LNt=pagemtimes(Lcol,Nt);
            dNxNt=pagemtimes(dNxC,Nt); dNyNt=pagemtimes(dNyC,Nt);

            Ke_tan(iu,iu,:)=Ke_tan(iu,iu,:)+dt*(NNt.*dudx_k+tau_s.*LNt.*dudx_k).*dV;
            Ke_tan(iu,iv,:)=Ke_tan(iu,iv,:)+dt*(NNt.*dudy_k+tau_s.*LNt.*dudy_k).*dV;
            Ke_tan(iv,iu,:)=Ke_tan(iv,iu,:)+dt*(NNt.*dvdx_k+tau_s.*LNt.*dvdx_k).*dV;
            Ke_tan(iv,iv,:)=Ke_tan(iv,iv,:)+dt*(NNt.*dvdy_k+tau_s.*LNt.*dvdy_k).*dV;
            Ke_tan(ip,iu,:)=Ke_tan(ip,iu,:)+dt*tau_s.*(dNxNt.*dudx_k+dNyNt.*dvdx_k).*dV;
            Ke_tan(ip,iv,:)=Ke_tan(ip,iv,:)+dt*tau_s.*(dNxNt.*dudy_k+dNyNt.*dvdy_k).*dV;
        end

        Uk_3d=reshape(Uk_l,ndof_e,1,ne);
        Re_vec=pagemtimes(Ke_pic,Uk_3d)-Fe_pic; Je_mat=Ke_pic+Ke_tan;
        r0=repmat((1:ndof_e)',1,ndof_e); c0=repmat(1:ndof_e,ndof_e,1); r0=r0(:); c0=c0(:);
        I_t=gather(edofs_g(:,r0)'); J_t=gather(edofs_g(:,c0)');
        V_t=gather(reshape(Je_mat,ndof_e^2,ne));
        I_all=[I_all;I_t(:)]; J_all=[J_all;J_t(:)]; V_all=[V_all;V_t(:)];
        ed_cpu=gather(edofs_g'); re_cpu=gather(reshape(Re_vec,ndof_e,ne));
        R_cpu=R_cpu+accumarray(ed_cpu(:),re_cpu(:),[total_dof,1]);
    end
    I=I_all; J_idx=J_all; V=V_all; R_global=R_cpu;
end