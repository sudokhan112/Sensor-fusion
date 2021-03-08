% 
weed_data = readtable('sensor data location');
time_table = weed_data(:,1);

time = table2array(time_table);
cocklebur = table2array(weed_data(:,2));
pigweed = table2array(weed_data(:,3));
ragweed = table2array(weed_data(:,4));

% plot(time,cocklebur,'g',time,pigweed,'r',time,ragweed,'b')
% title('Ragweed Pigweed separately placed')
% xlabel('time')
% ylabel('classification %')
% legend('cocklebur','ragweed','pigweed')

M = [cocklebur ragweed pigweed];

% M = [.6 .1 .3;
%       .65  .15  .2;
%       .6 .2 .2;
%       0 .85 .15;
%       .55 .2 .25;
%       .6  .2  .2;
%       .7  .1  .2];

ts = 10; %5 time steps
object = 3; % number of object to be classified
D_jos = [1 0 0;
         0 1 0;
         0 0 1];
total_time_steps = size(M,1); %number of row of M

for count = 1 : (total_time_steps - ts + 1)
%count = 3;    
for row_d = count : ts+count-1
    for column_d = count : ts+count-1
        D(row_d-count+1,column_d-count+1) = sqrt (.5 * (M(row_d,:) - M(column_d,:)) * D_jos * transpose(M(row_d,:) - M(column_d,:)));
    end
    d(row_d-count+1) = sum(D(row_d-count+1,:));
end
d_avg = sum(d)/length(d);

    %entropy_m1 = - (.6* log2(.6)+ .1* log2(.1)+ .3* log2(.3));
for e_row = count : ts+count-1
   entropy(e_row-count+1) = 0;
   for col = 1:object
       if (M(e_row,col)==0)
            ;
       else
            entropy(e_row-count+1) = entropy(e_row-count+1) + (-(M(e_row,col)*log2(M(e_row,col))));
       end  
    end
end
norm_entropy = entropy./sum(entropy);

    for row = 1:ts
        if (d(row) <= d_avg)
            reward(row) = -log(norm_entropy(row));
        else
            reward(row) = -log(1 - norm_entropy(row)); %penalty
        end
    end
    
    weight = reward./sum(reward);

    for i = 1:object
        evidence(i) = weight * M(count:ts+count-1,i);
    end


    evidence_new = evidence;

    for fus = 1:ts-1
        for row = 1:object
            for col = 1:object
                fusion(row,col,fus) = evidence(row)*evidence_new(:,col,fus);
            end
        end
        k(fus) = fusion(1,2,fus)+fusion(1,3,fus)+fusion(2,1,fus)+fusion(2,3,fus)+fusion(3,1,fus)+fusion(3,2,fus);
        den(fus) = 1 - k(fus);
        evidence_new(:,:,fus+1) = [fusion(1,1,fus) fusion(2,2,fus) fusion(3,3,fus)]./den(fus);
    end
    fused_evidence(:,:,count) = evidence_new;

end
orig_time_steps = 1:total_time_steps;
time_steps = ts:total_time_steps;

fused_cocklebur = squeeze(fused_evidence(1,ts,:));
fused_ragweed = squeeze(fused_evidence(2,ts,:));
fused_pigweed = squeeze(fused_evidence(3,ts,:));


subplot(2,1,1)
plot(orig_time_steps,cocklebur*100,'--g',orig_time_steps,pigweed*100,'--r',orig_time_steps,ragweed*100,'--b')
xlabel('time steps')
ylabel('classification %')
ylim([0 100])
title('Ragweed Pigweed separately placed: original')
legend('cocklebur','ragweed','pigweed')

subplot(2,1,2)
plot(time_steps,fused_cocklebur*100,'g',time_steps,fused_pigweed*100,'r',time_steps,fused_ragweed*100,'b')
title('Ragweed Pigweed separately placed: fused')
xlabel('time steps')
ylabel('classification %')
ylim([0 100])
legend('fused cocklebur','fused ragweed','fused pigweed')

%fused_ragweed_ts_5 = fused_ragweed;
%fused_ragweed_ts_3 = fused_ragweed;
%fused_ragweed_ts_15 = fused_ragweed;
