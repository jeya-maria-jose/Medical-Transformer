
% close all;
% clear all;
% clc;
N = 328
st = 0;
Fsc=[];
MIU=[];
PA=[];
bestfsc=0;
bestmiu=0;
bestpa=0;
bestep = 0;

for k = 0:8
    k
    Fsc=[];
    MIU=[];
    PA=[];
for i = st:st+N
    i;
   %gname = strcat('./Brain_test/',num2str(i,'%04d'),'.png');
   
   tname = '/media/jeyamariajose/7888230b-5c10-4229-90f2-c78bdae9c5de/Data/Projects/axialseg/KiU-Net-pytorch/results/brainus/mix_3_gated_wopos/';
   imgname = strcat(tname,num2str(50*k),'/',num2str(i,'%04d'),'.png');
   lname = '/media/jeyamariajose/7888230b-5c10-4229-90f2-c78bdae9c5de/Data/Brain_Ultrasound/Final/resized/test/labelcol/';
   labelname = strcat(lname, num2str(i,'%04d'),'.png');
   
   I = double(imread(imgname));tmp2=zeros(128,128);
   tmp2(I>131) = 255;
   tmp2(I<130) = 0;
   tmp = double(imread(labelname));
   tmp = tmp(:,:,1);
   tmp(tmp<130)=0;tmp(tmp>131)=255;
   
   tp=0;fp=0;fn=0;tn=0;uni=0;ttp=0;lab=0;
   
   for p =1:128
       for q =1:128
           if tmp(p,q)==0
               if tmp2(p,q) == tmp(p,q)
                   tn = tn+1;
               else
                   fp = fp+1;
                   uni = uni+1;
                   ttp = ttp+1;
               end
           elseif tmp(p,q)==255
               lab = lab +1;
               if tmp2(p,q) == tmp(p,q)
                   tp = tp+1;
                   ttp = ttp+1;
               else
                   fn = fn+1;
               end
               uni = uni+1;
           end
           
       end
   end
   
   if (tp~=0)
       F = (2*tp)/(2*tp+fp+fn);
       MIU=[MIU,(tp*1.0/uni)];
       PA=[PA,(tp*1.0/ttp)];
       Fsc=[Fsc;[i,F]];
   else
       MIU=[MIU,1];
       PA=[PA,1];
       Fsc=[Fsc;[i,1]];
   
   end
   

   
end
   if bestfsc <= mean(Fsc) & (mean(Fsc) ~= 1)
   bestfsc = mean(Fsc);
   bestmiu = mean(MIU,2);
   bestpa = mean(PA,2);
   bestep = 50*k;
   
   end
   mean(Fsc)
end

bestfsc
bestmiu
bestpa
bestep

% plot(Fsc(:,1),Fsc(:,2),'-*')
% hold on
% plot(Fsc(:,1),Fsc1(:,2),'-s')
% hold off
% figure();plot(Fsc(:,1),PA,'-*');hold on
% plot(Fsc(:,1),PA1,'-s');hold off
% Fsc1=Fsc;
% MIU1=MIU;
% PA1=PA;
