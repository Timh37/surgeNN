clc, clear all, close all

addpath('/Users/timhermans/Documents/MATLAB/t_tide_v1.5beta/')
%% Get list of file names
in_dir = '/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/detide_GESLA3/GESLA3_hourly/';
out_dir = '/Users/timhermans/Documents/PostDoc/Phase3_statistical_surges/Analysis/detide_GESLA3/detided_GESLA3_hourly/t_tide_results/';
cd(in_dir);

% Get list with file names
list = dir('*.csv');
list = list(3)
%% Run loop for tidal analysis with T_Tide for all sites seperately

for ii = 1:length(list)
    fname = list(ii).name;
    disp(fname)
    data = readtable(fname);

    timesteps = datenum(table2array(data(:,2)));
    surge = table2array(data(:,3));
    lat = table2array(data(1,4));
    
    %Pass on to Tide_analysis
    [ TCa, TCae, TCp, TCpe, Mt, tidal_prediction, nameu, years] = Tide_analysis_t_tide(timesteps,surge,lat);
    %a: amplitude, p: phase, e: confidence intervals, nameu: name of
    %constituents

    surge_nontidal = surge-tidal_prediction;
    
    output_data = data;
    output_data(:,3) = array2table(surge_nontidal);

       
    con_ampl = array2table(TCa,'VariableNames',cellstr(nameu));
    con_phase = array2table(TCp,'VariableNames',cellstr(nameu));
    con_ampl.year = years;
    con_phase.year = years;

    con_ampl = con_ampl(:,[end,1:end-1]);
    con_phase = con_phase(:,[end,1:end-1]);
    
    writetable(output_data,fullfile(out_dir,'detided_GESLA3_hourly',fname)); %store nontidal residuals
    writetable(con_ampl,fullfile(out_dir,'constituents_hourly','amplitude',fname)) %store constituent phase & amplitude
    writetable(con_phase,fullfile(out_dir,'constituents_hourly','phase',fname))

end