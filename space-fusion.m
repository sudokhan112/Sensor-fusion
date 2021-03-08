##% Step 1: Information matrix
%     A   B   C  A,C
M = [.41 .29 .3  0;
      0  .9  .1  0;
     .58 .07  0 .35;
     .55 .1   0 .35;
     .6  .1   0 .3];
  
 % Step 2: Distance matrix

 % D for jousselme distance 
 % DM = sqrt(.5 * (mi - mj)T * D * (mi - mj))
 % D :: {A} {B} {C} {A,C}
##   {A}
##   {B}
##   {C}
##   {A,C}
 D_jos = [1 0 0 .5;
          0 1 0  0;
          0 0 1 .5;
         .5 0 .5 1];
 
          
D12 = sqrt (.5 * (M(1,:) - M(2,:)) * D_jos * transpose(M(1,:) - M(2,:)));
D13 = sqrt (.5 * (M(1,:) - M(3,:)) * D_jos * transpose(M(1,:) - M(3,:)));
D14 = sqrt (.5 * (M(1,:) - M(4,:)) * D_jos * transpose(M(1,:) - M(4,:)));
D15 = sqrt (.5 * (M(1,:) - M(5,:)) * D_jos * transpose(M(1,:) - M(5,:)));
D23 = sqrt (.5 * (M(2,:) - M(3,:)) * D_jos * transpose(M(2,:) - M(3,:)));
D24 = sqrt (.5 * (M(2,:) - M(4,:)) * D_jos * transpose(M(2,:) - M(4,:)));
D25 = sqrt (.5 * (M(2,:) - M(5,:)) * D_jos * transpose(M(2,:) - M(5,:)));
D34 = sqrt (.5 * (M(3,:) - M(4,:)) * D_jos * transpose(M(3,:) - M(4,:)));
D35 = sqrt (.5 * (M(3,:) - M(5,:)) * D_jos * transpose(M(3,:) - M(5,:)));
D45 = sqrt (.5 * (M(4,:) - M(5,:)) * D_jos * transpose(M(4,:) - M(5,:)));

DM = [0   D12  D13   D14   D15;
     D12   0   D23   D24   D25;
     D13  D23   0    D34   D35;
     D14  D24  D34    0    D45;
     D15  D25  D35   D45    0 ];

%Step 3 calculate average evidence distance 
d1 = sum(DM(1,:));
d2 = sum(DM(2,:));
d3 = sum(DM(3,:));
d4 = sum(DM(4,:));
d5 = sum(DM(5,:));

%Step 4  calculate global average evidence distance 
d = (d1+d2+d3+d4+d5)/5;

%Step 5 Calculate belief entropy and normalize 
% Bel and Pl values
%       A     B       C                  A,C
% m1:  .41   .29     .3                   0
% m2:   0    .9      .1                   0
% m3:  .93   .07  (Bel=0,Pl=.35)         .93
% m4:  .9    .1   (Bel=0,Pl=.35)         .9
% m5:  .9    .1   (Bel=0,Pl=.3)          .9
entropy_m1 = - (.41* log2(.41)+ .29* log2(.29)+ .3* log2(.3));
entropy_m2 = - (.9* log2(.9)+ .1* log2(.1));
entropy_m3 = - (.93* log2(.93)+ .07* log2(.07)+(.35/2)*log2(.35/2)+ .93* log2((.93/2)*exp(2/3)));
entropy_m4 = - (.9* log2(.9)+ .1* log2(.1)+(.35/2)*log2(.35/2)+ .9* log2((.9/2)*exp(2/3)));
entropy_m5 = - (.9* log2(.9)+ .1* log2(.1)+(.3/2)*log2(.3/2)+ .9* log2((.9/2)*exp(2/3)));
%normalize entropy
norm_entropy_m1 = entropy_m1/(entropy_m1+entropy_m2+entropy_m3+entropy_m4+entropy_m5);
norm_entropy_m2 = entropy_m2/(entropy_m1+entropy_m2+entropy_m3+entropy_m4+entropy_m5);
norm_entropy_m3 = entropy_m3/(entropy_m1+entropy_m2+entropy_m3+entropy_m4+entropy_m5);
norm_entropy_m4 = entropy_m4/(entropy_m1+entropy_m2+entropy_m3+entropy_m4+entropy_m5);
norm_entropy_m5 = entropy_m5/(entropy_m1+entropy_m2+entropy_m3+entropy_m4+entropy_m5);

%step 6 Normalize evidence Reward and Penalty to get evidence weight.
reward_1 = -log(norm_entropy_m1); %d1>d
penalty_2 = -log(1-norm_entropy_m2);%d2<d
reward_3 = -log(norm_entropy_m3); %d3>d
reward_4 = -log(norm_entropy_m4); %d4>d
reward_5 = -log(norm_entropy_m5); %d5>d

w1 = reward_1/(reward_1+penalty_2+reward_3+reward_4+reward_5);
w2 = penalty_2/(reward_1+penalty_2+reward_3+reward_4+reward_5);
w3 = reward_3/(reward_1+penalty_2+reward_3+reward_4+reward_5);
w4 = reward_4/(reward_1+penalty_2+reward_3+reward_4+reward_5);
w5 = reward_5/(reward_1+penalty_2+reward_3+reward_4+reward_5);


%Step 7: Modify the original evidence
weight = [w1 w2 w3 w4 w5]

m_A = weight * M(:,1);
m_B = weight * M(:,2);
m_C = weight * M(:,3);
m_A_C = weight * M(:,4);

%step 8 combine for (n-1) times with DS combination rule
%           m(A)   m(B)    m(c)   m(A,C)

% m(A)      m(A)   k       k       m(A)
% m(B)      k      m(B)    k        k
% m(C)      k      k      m(C)     m(C)
% m(A,C)   m(A)    k      m(C)     m(A,C)

%fusion 1-2
m1 = [m_A m_B m_C m_A_C]
%m1_new = M(2,:); %sensor 2

fus12 = [m1(1,1)*m1(1,1) m1(1,1)*m1(1,2) m1(1,1)*m1(1,3) m1(1,1)*m1(1,4);
         m1(1,2)*m1(1,1) m1(1,2)*m1(1,2) m1(1,2)*m1(1,3) m1(1,2)*m1(1,4);
         m1(1,3)*m1(1,1) m1(1,3)*m1(1,2) m1(1,3)*m1(1,3) m1(1,3)*m1(1,4);
         m1(1,4)*m1(1,1) m1(1,4)*m1(1,2) m1(1,4)*m1(1,3) m1(1,4)*m1(1,4)];
         
##fus12 = [m1(1,1)*m1_new(1,1) m1(1,1)*m1_new(1,2) m1(1,1)*m1_new(1,3) m1(1,1)*m1_new(1,4);
##         m1(1,2)*m1_new(1,1) m1(1,2)*m1_new(1,2) m1(1,2)*m1_new(1,3) m1(1,2)*m1_new(1,4);
##         m1(1,3)*m1_new(1,1) m1(1,3)*m1_new(1,2) m1(1,3)*m1_new(1,3) m1(1,3)*m1_new(1,4);
##         m1(1,4)*m1_new(1,1) m1(1,4)*m1_new(1,2) m1(1,4)*m1_new(1,3) m1(1,4)*m1_new(1,4)];

         
k12 = fus12(1,2)+fus12(1,3)+fus12(2,1)+fus12(2,3)+fus12(2,4)+fus12(3,1)+fus12(3,2)+fus12(4,2);
den12 = 1-k12;
m12_A = (fus12(1,1)+fus12(1,4)+fus12(4,1))/den12;
m12_B = (fus12(2,2))/den12;
m12_C = (fus12(3,3)+fus12(3,4)+fus12(4,3))/den12;
m12_A_C = (fus12(4,4))/den12;

%fusion 1-2-3
printf("sensor 1-2 fusion")
m12 = [m12_A m12_B m12_C m12_A_C]
%m1_new = M(3,:); %sensor3

fus123 = [m12(1,1)*m1(1,1) m12(1,1)*m1(1,2) m12(1,1)*m1(1,3) m12(1,1)*m1(1,4);
         m12(1,2)*m1(1,1) m12(1,2)*m1(1,2) m12(1,2)*m1(1,3) m12(1,2)*m1(1,4);
         m12(1,3)*m1(1,1) m12(1,3)*m1(1,2) m12(1,3)*m1(1,3) m12(1,3)*m1(1,4);
         m12(1,4)*m1(1,1) m12(1,4)*m1(1,2) m12(1,4)*m1(1,3) m12(1,4)*m1(1,4)];

##fus123 = [m12(1,1)*m1_new(1,1) m12(1,1)*m1_new(1,2) m12(1,1)*m1_new(1,3) m12(1,1)*m1_new(1,4);
##         m12(1,2)*m1_new(1,1) m12(1,2)*m1_new(1,2) m12(1,2)*m1_new(1,3) m12(1,2)*m1_new(1,4);
##         m12(1,3)*m1_new(1,1) m12(1,3)*m1_new(1,2) m12(1,3)*m1_new(1,3) m12(1,3)*m1_new(1,4);
##         m12(1,4)*m1_new(1,1) m12(1,4)*m1_new(1,2) m12(1,4)*m1_new(1,3) m12(1,4)*m1_new(1,4)];

##fus123 = [m1(1,1)*m1_new(1,1) m1(1,1)*m1_new(1,2) m1(1,1)*m1_new(1,3) m1(1,1)*m1_new(1,4);
##         m1(1,2)*m1_new(1,1) m1(1,2)*m1_new(1,2) m1(1,2)*m1_new(1,3) m1(1,2)*m1_new(1,4);
##         m1(1,3)*m1_new(1,1) m1(1,3)*m1_new(1,2) m1(1,3)*m1_new(1,3) m1(1,3)*m1_new(1,4);
##         m1(1,4)*m1_new(1,1) m1(1,4)*m1_new(1,2) m1(1,4)*m1_new(1,3) m1(1,4)*m1_new(1,4)];

k123 = fus123(1,2)+fus123(1,3)+fus123(2,1)+fus123(2,3)+fus123(2,4)+fus123(3,1)+fus123(3,2)+fus123(4,2);
den123 = 1-k123;
m123_A = (fus123(1,1)+fus123(1,4)+fus123(4,1))/den123;
m123_B = (fus123(2,2))/den123;
m123_C = (fus123(3,3)+fus123(3,4)+fus123(4,3))/den123;
m123_A_C = (fus123(4,4))/den123;

%fusion 1-2-3-4
printf("sensor 1-2-3 fusion")
m123 = [m123_A m123_B m123_C m123_A_C]
%m1_new = M(4,:); %sensor4

fus1234 = [m123(1,1)*m1(1,1) m123(1,1)*m1(1,2) m123(1,1)*m1(1,3) m123(1,1)*m1(1,4);
         m123(1,2)*m1(1,1) m123(1,2)*m1(1,2) m123(1,2)*m1(1,3) m123(1,2)*m1(1,4);
         m123(1,3)*m1(1,1) m123(1,3)*m1(1,2) m123(1,3)*m1(1,3) m123(1,3)*m1(1,4);
         m123(1,4)*m1(1,1) m123(1,4)*m1(1,2) m123(1,4)*m1(1,3) m123(1,4)*m1(1,4)];


##fus1234 = [m123(1,1)*m1_new(1,1) m123(1,1)*m1_new(1,2) m123(1,1)*m1_new(1,3) m123(1,1)*m1_new(1,4);
##         m123(1,2)*m1_new(1,1) m123(1,2)*m1_new(1,2) m123(1,2)*m1_new(1,3) m123(1,2)*m1_new(1,4);
##         m123(1,3)*m1_new(1,1) m123(1,3)*m1_new(1,2) m123(1,3)*m1_new(1,3) m123(1,3)*m1_new(1,4);
##         m123(1,4)*m1_new(1,1) m123(1,4)*m1_new(1,2) m123(1,4)*m1_new(1,3) m123(1,4)*m1_new(1,4)];

##fus1234 = [m1(1,1)*m1_new(1,1) m1(1,1)*m1_new(1,2) m1(1,1)*m1_new(1,3) m1(1,1)*m1_new(1,4);
##         m1(1,2)*m1_new(1,1) m1(1,2)*m1_new(1,2) m1(1,2)*m1_new(1,3) m1(1,2)*m1_new(1,4);
##         m1(1,3)*m1_new(1,1) m1(1,3)*m1_new(1,2) m1(1,3)*m1_new(1,3) m1(1,3)*m1_new(1,4);
##         m1(1,4)*m1_new(1,1) m1(1,4)*m1_new(1,2) m1(1,4)*m1_new(1,3) m1(1,4)*m1_new(1,4)];

k1234 = fus1234(1,2)+fus1234(1,3)+fus1234(2,1)+fus1234(2,3)+fus1234(2,4)+fus1234(3,1)+fus1234(3,2)+fus1234(4,2);
den1234 = 1-k1234;
m1234_A = (fus1234(1,1)+fus1234(1,4)+fus1234(4,1))/den1234;
m1234_B = (fus1234(2,2))/den1234;
m1234_C = (fus1234(3,3)+fus1234(3,4)+fus1234(4,3))/den1234;
m1234_A_C = (fus1234(4,4))/den1234;

%fusion 1-2-3-4-5
printf("sensor 1-2-3-4 fusion")
m1234 = [m1234_A m1234_B m1234_C m1234_A_C]
%m1_new = M(5,:); %sensor5

fus12345 = [m1234(1,1)*m1(1,1) m1234(1,1)*m1(1,2) m1234(1,1)*m1(1,3) m1234(1,1)*m1(1,4);
         m1234(1,2)*m1(1,1) m1234(1,2)*m1(1,2) m1234(1,2)*m1(1,3) m1234(1,2)*m1(1,4);
         m1234(1,3)*m1(1,1) m1234(1,3)*m1(1,2) m1234(1,3)*m1(1,3) m1234(1,3)*m1(1,4);
         m1234(1,4)*m1(1,1) m1234(1,4)*m1(1,2) m1234(1,4)*m1(1,3) m1234(1,4)*m1(1,4)];

##fus12345 = [m1234(1,1)*m1_new(1,1) m1234(1,1)*m1_new(1,2) m1234(1,1)*m1_new(1,3) m1234(1,1)*m1_new(1,4);
##         m1234(1,2)*m1_new(1,1) m1234(1,2)*m1_new(1,2) m1234(1,2)*m1_new(1,3) m1234(1,2)*m1_new(1,4);
##         m1234(1,3)*m1_new(1,1) m1234(1,3)*m1_new(1,2) m1234(1,3)*m1_new(1,3) m1234(1,3)*m1_new(1,4);
##         m1234(1,4)*m1_new(1,1) m1234(1,4)*m1_new(1,2) m1234(1,4)*m1_new(1,3) m1234(1,4)*m1_new(1,4)];

##fus12345 = [m1(1,1)*m1_new(1,1) m1(1,1)*m1_new(1,2) m1(1,1)*m1_new(1,3) m1(1,1)*m1_new(1,4);
##         m1(1,2)*m1_new(1,1) m1(1,2)*m1_new(1,2) m1(1,2)*m1_new(1,3) m1(1,2)*m1_new(1,4);
##         m1(1,3)*m1_new(1,1) m1(1,3)*m1_new(1,2) m1(1,3)*m1_new(1,3) m1(1,3)*m1_new(1,4);
##         m1(1,4)*m1_new(1,1) m1(1,4)*m1_new(1,2) m1(1,4)*m1_new(1,3) m1(1,4)*m1_new(1,4)];

k12345 = fus12345(1,2)+fus12345(1,3)+fus12345(2,1)+fus12345(2,3)+fus12345(2,4)+fus12345(3,1)+fus12345(3,2)+fus12345(4,2);
den12345 = 1-k12345;
m12345_A = (fus12345(1,1)+fus12345(1,4)+fus12345(4,1))/den12345;
m12345_B = (fus12345(2,2))/den12345;
m12345_C = (fus12345(3,3)+fus12345(3,4)+fus12345(4,3))/den12345;
m12345_A_C = (fus12345(4,4))/den12345;

printf("sensor 1-2-3-4-5 fusion")
m12345 = [m12345_A m12345_B m12345_C m12345_A_C]



















