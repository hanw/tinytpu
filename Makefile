BSC = bsc -no-show-timestamps -no-show-version
BSCFLAGS = -sim -p +

# Compile any .bsv to .ba
%.bo: %.bsv
	$(BSC) $(BSCFLAGS) $<

# PE testbench
mkTbPE.bexe: TbPE.bo PE.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbPE mkTbPE.ba

test-pe: mkTbPE.bexe
	./$<

# Array testbench
mkTbSystolicArray.bexe: TbSystolicArray.bo SystolicArray.bo PE.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbSystolicArray mkTbSystolicArray.ba

test-array: mkTbSystolicArray.bexe
	./$<

# Full accelerator testbench
mkTbAccelerator.bexe: TbAccelerator.bo TensorAccelerator.bo Controller.bo SystolicArray.bo PE.bo WeightSRAM.bo ActivationSRAM.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbAccelerator mkTbAccelerator.ba

test-accel: mkTbAccelerator.bexe
	./$<

# 4x4 testbench
mkTbAccelerator4x4.bexe: TbAccelerator4x4.bo TensorAccelerator.bo Controller.bo SystolicArray.bo PE.bo WeightSRAM.bo ActivationSRAM.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbAccelerator4x4 mkTbAccelerator4x4.ba

test-4x4: mkTbAccelerator4x4.bexe
	./$<

test-all: test-pe test-array test-accel test-4x4

# Inter-package dependencies
TbPE.bo: PE.bo
TbSystolicArray.bo: SystolicArray.bo
SystolicArray.bo: PE.bo
Controller.bo: SystolicArray.bo WeightSRAM.bo ActivationSRAM.bo
TensorAccelerator.bo: SystolicArray.bo WeightSRAM.bo ActivationSRAM.bo Controller.bo
TbAccelerator.bo: TensorAccelerator.bo
TbAccelerator4x4.bo: TensorAccelerator.bo

.PHONY: clean test-pe test-array test-accel test-4x4 test-all
clean:
	rm -f *.bi *.bo *.ba *.bexe *.cxx *.h *.o *.so
