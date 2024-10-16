% TMz Simple code
clear
%Constants

mo=400*pi*1e-9;
co=2.997925e8;
eo=1.0/(mo*(co)^2);
Zo=sqrt(mo/eo);

%Model params

NX=200; %Domain size in X in cells
NY=200; %Domain size in Y in cells

Iter=415; %Number of iterations, total time is Iter*dt

dx=0.001; %Spacial step in x
dy=0.001; %Spacial step in y

dt=1/(co*sqrt(1.0/(dx^2) + 1.0/(dy^2))); %time step determined by the CFL stability condition

disp(['Velocity of Light [m/s] ' num2str(co)]);
disp(['Permittivity of Free space [F/m] '  num2str(eo)]);
disp(['Permeability of Free space [H/m] '  num2str(mo)]);
disp(['Impedance of Free space [Ohms] ' num2str(Zo)]);

disp(['FDTD Model Space in cells ' num2str(NX) ', ' num2str(NY)]);
disp(['Number of Iterations ' num2str(Iter)]);
disp(['Dx = ' num2str(dx) ',Dy = ' num2str(dy) ',Dt = ' num2str(dt)]);

Ez=zeros(NX,NY); %Ez field array
Hx=zeros(NX,NY+1); %Hx field array
Hy=zeros(NX+1,NY); %Hy field array

src=zeros(1,Iter+1); %vector to hold the excitation function

Sx = 50; %X coordinate for the source - change this for other places
Sy = 50; %Y coordiante for the source - change this for other places

Ox = 100; %X coordinate for the monitor (solution) - change this for other places
Oy = 150; %Y coordiante for the monitor (solution) - change this for other places

Out_Ez = zeros(1,Iter+1); %vector to hold output point Hz component
Out_Hx = zeros(1,Iter+1); %vector to hold output point Ex component
Out_Hy = zeros(1,Iter+1); %vector to hold output point Ey component

%destination for vectoried form
Out_vEz=zeros(1,Iter+1);
Out_vHx=zeros(1,Iter+1);
Out_vHy=zeros(1,Iter+1);


% Assume non-magnetic media and use only relative pernittivity and
% conductivity for targets

ER = ones(NX,NY)*eo; %Free Space 
SIGMA = zeros(NX,NY); %zero conductivity 

%build a cylinder. Use copy/paste to build more of these.
Radius = 20;
CenterX = 110;
CenterY = 110;
relative_permittivity = 4.0;
sigma = 0.1;

for i=CenterX-Radius-1:CenterX+Radius+1
    for j=CenterY-Radius-1:CenterY+Radius+1

        if sqrt((i-CenterX).^2+(j-CenterY).^2)<=Radius 
            ER(i,j)=relative_permittivity*eo;
            SIGMA(i,j) = sigma;
        end
    end
end

%end building objects

DA = (2*ER - SIGMA*dt)./(2*ER + SIGMA*dt);
CA = (2*dt./(2*ER + SIGMA*dt));

%Plot the electrical properties of the domain
figure; subplot(1,2,1); imagesc(ER); axis square; colorbar horr; 
subtitle('Relative permittivity');
subplot(1,2,2); imagesc(SIGMA); axis square; colorbar horr;
subtitle('Conductivity');
disp('Press enter to see the simulation')
pause;
close;


%Vectorise
vDA=zeros(NX*NY,1);
vCA=zeros(NX*NY,1);

for i=1:NX
    for j=1:NY
        vDA(j+(i-1)*NY) = DA(i,j); 
        vCA(j+(i-1)*NY) = CA(i,j);
    end
end


DB=1;
CB=dt/mo;


%source definition - can try other things ...
tw=26.53e-12;
to=4*tw;

for i=1:Iter+1
	t=i*dt;
	src(i)=-2.0*((t-to)/tw)*exp(-((t-to)/tw)^2);
end


ind=1;
time = 0;

%Start the time iterations
while ind<Iter	


for i=1:NX
    for j=2:NY
		Hx(i,j)=DB*Hx(i,j) - CB*(Ez(i,j) - Ez(i,j-1))/dy;
   end
end


for i=2:NX 
    for j=1:NY
 		Hy(i,j)=DB*Hy(i,j) + CB*(Ez(i,j) - Ez(i-1,j))/dx; 		
    end
end


time=time+dt*0.5;

for i=1:NX
    for j=1:NY
		Ez(i,j)=DA(i,j)*Ez(i,j) + CA(i,j)*(Hy(i+1,j) - Hy(i,j))/dx - CA(i,j)*(Hx(i,j+1) - Hx(i,j))/dy;
    end
end

%Adjust Ez at the source
Ez(Sx,Sy)=Ez(Sx,Sy) - CA(Sx,Sy)*src(ind)*(1/(dx*dy));


time=time+dt*05;  


%Plot the fields
subplot(1,3,1); cla; imagesc(Ez); axis square; colorbar horr; subtitle('Ez');
subplot(1,3,2); cla; imagesc(Hx); axis square; colorbar horr; subtitle('Hx');
subplot(1,3,3); cla; imagesc(Hy); axis square; colorbar horr; subtitle('Hy');
drawnow;
pause(0.1);


ind=ind+1;

end

disp('Done!');