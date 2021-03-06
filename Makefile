cuda: cuda/*
	nvcc -g cuda/*.cu -Xptxas -O3 -use_fast_math -arch=sm_61  --relocatable-device-code=true -o run  --expt-relaxed-constexpr -I cuda -lglfw -lGLEW -lGL;
	./run

debug: cuda/*
	nvcc cuda/*.cu -Xptxas -O1 -g -use_fast_math -arch=sm_61  --relocatable-device-code=true -o run  --expt-relaxed-constexpr -I cuda -lglfw -lGLEW -lGL;
	./run