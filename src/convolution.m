load ./kernel.mat
load ./image.mat

mk = 5;
nk = 5;
filter = zeros(mk, nk);

for c = 1, 3
  filter += conv2(image(:,:,c), kernel(:,:,c), "valid");
end
disp(filter);