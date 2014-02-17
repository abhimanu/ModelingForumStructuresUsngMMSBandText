function [] = viz_stanford(thetaU, startIndx, endIndx)
% Arranges vertices from the Stanford graph in the form of a pentagon

% thetaU=load('finalAnalysis/llPlots/SO_Thresh1_6Oct_combined.pidegree');
K=5;
theta = thetaU(:,startIndx:endIndx);
% [coeff,theta] = pca(thetaU(:,3:end),'NumComponents',K);    % remove the used id
% % Generate theta
% theta = double(data.samples{end}.stheta) + data.samples{end}.alpha;
% theta = theta ./ repmat(sum(theta,2),[1 size(theta,2)]);

% Generate pentagon endpoints
size(theta)
pentagon_points = zeros(K,2);
for i = 1:K
  pentagon_points(i,1) = sin(2*pi*(i-1)/K);
  pentagon_points(i,2) = cos(2*pi*(i-1)/K);
end

% Generate colors
colors = lines(K);

% Compute theta positions and colors
theta_pentagon_positions = theta * pentagon_points;
theta_colors = theta * colors;

% Compute theta sizes according to degree
% node_degrees = load('SO_Thresh1_6Oct_combined.degree');
theta_sizes = 1 + thetaU(:,2);

% Sort all theta information in descending order of theta_size, to ensure scatter() plots larger circles first
[~,idx] = sort(theta_sizes,'descend');
theta_pentagon_positions = theta_pentagon_positions(idx,:);
theta_colors = theta_colors(idx,:);
theta_sizes = theta_sizes(idx,:);

% Visualize thetas
scatter(theta_pentagon_positions(:,1),theta_pentagon_positions(:,2),theta_sizes,theta_colors,'o','filled');
axis off;

end