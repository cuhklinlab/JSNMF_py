% example main_function
% for pbmc data
load('pbmc_5k_10k.mat')
housekeeping_genes = ["ACTB";'ALDOA';'GAPDH';'PGK1';'LDHA';'RPS27A';'RPL19';'RPL11';'NONO';'ARHGDIA';'RPL32';'RPS18';'HSPCB';'C1orf43';'CHMP2A';'EMC7';'GPI';'PSMB2';'PSMB4';'RAB7A';'REEP5';'SNRPD3';'VCP';'VPS29'];
% List of Marker Genes
marker_genes = ["CD209";'ENG'; 'FOXP3'; 'CD34'; 'BATF3'; 'S100A12'; 'THBD';'CD3D'; 'THY1'; 'CD8A'; 'CD8B'; 'CD14'; 'PROM1'; 'IL2RA'; 'FCGR3A';'IL3RA'; 'FCGR1A'; 'CD19'; 'IL7R'; 'CD79A'; 'MS4A1'; 'NCAM1';'CD3E'; 'CD3G'; 'KIT'; 'CD1C'; 'CD68'; 'CD4'];
% using normalized rna data
load('processed.mat', 'RNA')
genes = RNA.Features; data = RNA.data;
sM = sum(data,2); zero_row = find(sM==0);
data(zero_row,:) = []; genes(zero_row) = [];

[~,~,ind1] = intersect(marker_genes,genes,'stable');
[~,~,ind2] = intersect(housekeeping_genes,genes,'stable');
clear RNA celltype genes housekeeping_genes marker_genes
num_clu = length(unique(label));

% run jsnmf
[alpha,gamma,Inits] = parameter_selection(X1,X2,label);
tic
[W1,W2,H1,H2,S,iter,objs] = gcnmf_pbmc(X1,X2,alpha,1,gamma,Inits); % beta=1;
disp('JSNMF runtime:');
toc

[clust,~,~] = getNCluster(S,num_clu,0,3,20); 
if length(unique(clust))== num_clu
   [ac, nmi_value, ~] = CalcMetrics(label, clust);
else
   [clust, ~] = SpectralClustering(S, num_clu);
   [ac, nmi_value, ~] = CalcMetrics(label, clust);   
end  
[ave_mk_gini, ave_hk_gini, difgini] = RAGI(data,ind1,ind2,clust);

%% for skin data
load('skin/skin_5k_10k.mat');
[alpha,gamma,Inits] = parameter_selection(X1,X2,label);
tic
 [W1,W2,H1,H2,S,iter,objs] = gcnmf_pbmc(X1,X2,alpha,1,gamma,Inits); % beta=1;
disp('JSNMF runtime:');
toc

[clust,~,~] = getNCluster(S,num_clu,0,3,20); 
if length(unique(clust))== num_clu
   [ac, nmi_value, ~] = CalcMetrics(label, clust); 
else
   [clust, ~] = SpectralClustering(S, num_clu);
   [ac, nmi_value, ~] = CalcMetrics(label, clust);   
end  










