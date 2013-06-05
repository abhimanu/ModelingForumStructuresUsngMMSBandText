function [] = plotSimplex(A,v)

if nargin==2
    
    figure
    % Plot the ternary2axis system
    [h,hg,htick]=terplot;
    % Plot the data
    % First set the colormap (can't be done afterwards)
    colormap(jet)
    [hd,hcb]=ternaryc(A(:,1),A(:,2),A(:,3),v,'o');
    % Add the labels
    hlabels=terlabel('A1','A2','A3');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The following modifications are not serious, just to illustrate how to
    % use the handles:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %--  Change the color of the grid lines
    set(hg(:,3),'color','m')
    set(hg(:,2),'color','c')
    set(hg(:,1),'color','y')
    %--  Change the marker size
    set(hd,'markersize',3)
    %--  Modify the labels
    set(hlabels,'fontsize',12)
    set(hlabels(3),'color','m')
    set(hlabels(2),'color','c')
    set(hlabels(1),'color','y')
    %--  Modify the tick labels
    set(htick(:,1),'color','y','linewidth',3)
    set(htick(:,2),'color','c','linewidth',3)
    set(htick(:,3),'color','m','linewidth',3)
    %--  Change the color of the patch
    set(h,'facecolor',[0.7 0.7 0.7],'edgecolor','w')
    %--  Change the colorbar
    set(hcb,'xcolor','w','ycolor','w')
    %--  Modify the figure color
    set(gcf,'color',[0 0 0.3])
    %-- Change some defaults
    set(gcf,'paperpositionmode','auto','inverthardcopy','off')
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lastly, an example showing the "constant data option" of
% ternaryc().
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

else
    figure
    %-- Plot the axis system
    [h,hg,htick]=terplot;
    %-- Plot the data ...
    hter=ternaryc(A(:,1),A(:,2),A(:,3));
    %-- ... and modify the symbol:
    set(hter,'marker','o','markerfacecolor','none','markersize',4)
    hlabels=terlabel('A1','A2','A3');
end
end