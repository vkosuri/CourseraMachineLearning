% Machine learning class 
% Octave tutorial 

% =======================================================
% Section 1: Octave Tutorial: Basic operations

%% Change Octave prompt  
PS1('>> ');              

%% elementary operations
5+6
3-2
5*8
1/2
2^6
1 == 2  % false
1 ~= 2  % true.  note, not "!="
1 && 0
1 || 0
xor(1,0)


%% variable assignment
a = 3; % semicolon suppresses output
b = 'hi';
c = 3>=1;

% Displaying them:
a = pi
disp(sprintf('2 decimals: %0.2f', a))
disp(sprintf('6 decimals: %0.6f', a))
format long
a
format short
a


%%  vectors and matrices
A = [1 2; 3 4; 5 6]

v = [1 2 3]
v = [1; 2; 3]
v = [1:0.1:2]  % from 1 to 2, with stepsize of 0.1. Useful for plot axes
v = 1:6        % from 1 to 6, assumes stepsize of 1

C = 2*ones(2,3)  % same as C = [2 2 2; 2 2 2]
w = ones(1,3)    % 1x3 vector of ones
w = zeros(1,3)
w = rand(1,3)  % drawn from a uniform distribution 
w = randn(1,3) % drawn from a normal distribution (mean=0, var=1)
w = -6 + sqrt(10)*(randn(1,10000))  % (mean = 1, var = 2)
hist(w)
I = eye(4)    % 4x4 identity matrix

% help function
help eye
help rand

% =======================================================
% Section 2: Octave Tutorial: Moving data around 


%% dimensions
sz = size(A)
size(A,1)  % number of rows
size(A,2)  % number of cols
length(v)  % size of longest dimension


%% loading data
pwd    % show current directory (current path)
cd 'C:\Users\ang\Octave files'   % change directory 
ls     % list files in current directory 
load q1y.dat
load q1x.dat
who    % list variables in workspace
whos   % list variables in workspace (detailed view) 
clear q1y       % clear w/ no argt clears all
v = q1x(1:10);
save hello v;   % save variable v into file hello.mat
save hello.txt v -ascii; % save as ascii
% fopen, fread, fprintf, fscanf also work  [[not needed in class]]

%% indexing
A(3,2)  % indexing is (row,col)
A(2,:)  % get the 2nd row. 
        % ":" means every element along that dimension
A(:,2)  % get the 2nd col
A([1 3],:)

A(:,2) = [10; 11; 12]     % change second column
A = [A, [100; 101; 102]]; % append column vec
A(:) % Select all elements as a column vector.

% Putting data together 
A = [A [100; 101; 102]]
B = [11 12; 13 14; 15 16] % same dims as A
[A B]
[A; B] 


% =======================================================
% Section 3: Octave Tutorial: Computing on data 


%% matrix operations
A * C  % matrix multiplication
A .* B % element-wise multiplcation
% A .* C  or A * B gives error - wrong dimensions
A .^ 2
1./v
log(v)  % functions like this operate element-wise on vecs or matrices 
exp(v)  % e^4
abs(v)

-v  % -1*v

v + ones(1,length(v))
% v + 1  % same

A'  % matrix transpose

%% misc useful functions

% max  (or min)
a = [1 15 2 0.5]
val = max(a)
[val,ind] = max(a)

% find
a < 3
find(a < 3)
A = magic(3)
[r,c] = find(A>=7)

% sum, prod
sum(a)
prod(a)
floor(a) % or ceil(a)
max(rand(3),rand(3))
max(A,[],1)
min(A,[],2)
A = magic(9)
sum(A,1)
sum(A,2)
sum(sum( A .* eye(9) ))
sum(sum( A .* flipud(eye(9)) ))


% Matrix inverse (pseudo-inverse)
pinv(A)        % inv(A'*A)*A'


% =======================================================
% Section 4: Octave Tutorial: Plotting 


%% plotting
t = [0:0.01:0.98];
y1 = sin(2*pi*4*t); 
plot(t,y1);
y2 = cos(2*pi*4*t);
hold on;  % "hold off" to turn off
plot(t,y2,'r');
xlabel('time');
ylabel('value');
legend('sin','cos');
title('my plot');
print -dpng 'myPlot.png'
close;           % or,  "close all" to close all figs

figure(2), clf;  % can specify the figure number
subplot(1,2,1);  % Divide plot into 1x2 grid, access 1st element
plot(t,y1);
subplot(1,2,2);  % Divide plot into 1x2 grid, access 2nd element
plot(t,y2);
axis([0.5 1 -1 1]);  % change axis scale

%% display a matrix (or image) 
figure;
imagesc(magic(15)), colorbar, colormap gray;
% comma-chaining function calls.  
a=1,b=2,c=3
a=1;b=2;c=3;


% =======================================================
% Section 5: Octave Tutorial: For, while, if statements, and functions.

v = zeros(10,1);
for i=1:10, 
    v(i) = 2^i;
end
% Can also use "break" and "continue" inside for and while loops to control execution.

i = 1;
while i <= 5,
  v(i) = 100; 
  i = i+1;
end

i = 1;
while true, 
  v(i) = 999; 
  i = i+1;
  if i == 6,
    break;
  end;
end

if v(1)==1,
  disp('The value is one!');
elseif v(1)==2,
  disp('The value is two!');
else
  disp('The value is not one or two!');
end

% exit  % quit

% Functions

% Create a file called squareThisNumber.m with the following contents (without the %):
% function r = squareThisNumber(x)
%   r = x * x;
% end

squareThisNumber(5);  
% If function is undefine, use "pwd" to check current directory (path), 
% and "cd" to change directories
pwd
cd 'C:\Users\ang\Desktop';
squareThisNumber(5);  

% Octave search path (advanced/optional) 
addpath('C:\Users\ang\Desktop');
cd 'C:\'
squareThisNumber(5);

% If you have defined other functions such as costFunctionJ, 
% the following code will work too. 

X = [1 1; 1 2; 1 3];
y = [1;2;3];

theta = [0; 1]; 
j = costFunctionJ(X, y, theta);

theta = [0; 0]; 
j = costFunctionJ(X, y, theta);




