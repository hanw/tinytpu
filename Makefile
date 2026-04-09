BSC      = bsc -no-show-timestamps -no-show-version
BUILDDIR = build
BSCFLAGS = -sim -p src:test:+ -bdir $(BUILDDIR) -simdir $(BUILDDIR)

$(BUILDDIR):
	mkdir -p $@

# --- Compile rules ---

$(BUILDDIR)/%.bo: src/%.bsv | $(BUILDDIR)
	$(BSC) $(BSCFLAGS) $<

$(BUILDDIR)/%.bo: test/%.bsv | $(BUILDDIR)
	$(BSC) $(BSCFLAGS) $<

# --- Link rules ---

$(BUILDDIR)/mkTbPE.bexe: $(BUILDDIR)/TbPE.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbPE $(BUILDDIR)/mkTbPE.ba

$(BUILDDIR)/mkTbSystolicArray.bexe: $(BUILDDIR)/TbSystolicArray.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbSystolicArray $(BUILDDIR)/mkTbSystolicArray.ba

$(BUILDDIR)/mkTbAccelerator.bexe: $(BUILDDIR)/TbAccelerator.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbAccelerator $(BUILDDIR)/mkTbAccelerator.ba

$(BUILDDIR)/mkTbAccelerator4x4.bexe: $(BUILDDIR)/TbAccelerator4x4.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbAccelerator4x4 $(BUILDDIR)/mkTbAccelerator4x4.ba

$(BUILDDIR)/mkTbXLU.bexe: $(BUILDDIR)/TbXLU.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbXLU $(BUILDDIR)/mkTbXLU.ba

$(BUILDDIR)/mkTbVMEM.bexe: $(BUILDDIR)/TbVMEM.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbVMEM $(BUILDDIR)/mkTbVMEM.ba

$(BUILDDIR)/mkTbVRegFile.bexe: $(BUILDDIR)/TbVRegFile.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbVRegFile $(BUILDDIR)/mkTbVRegFile.ba

$(BUILDDIR)/mkTbVPU.bexe: $(BUILDDIR)/TbVPU.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbVPU $(BUILDDIR)/mkTbVPU.ba

$(BUILDDIR)/mkTbScalarUnit.bexe: $(BUILDDIR)/TbScalarUnit.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbScalarUnit $(BUILDDIR)/mkTbScalarUnit.ba

$(BUILDDIR)/mkTbTensorCore.bexe: $(BUILDDIR)/TbTensorCore.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbTensorCore $(BUILDDIR)/mkTbTensorCore.ba

$(BUILDDIR)/mkTbSparseCore.bexe: $(BUILDDIR)/TbSparseCore.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbSparseCore $(BUILDDIR)/mkTbSparseCore.ba

# --- Test targets ---

test-pe: $(BUILDDIR)/mkTbPE.bexe
	$<

test-array: $(BUILDDIR)/mkTbSystolicArray.bexe
	$<

test-accel: $(BUILDDIR)/mkTbAccelerator.bexe
	$<

test-4x4: $(BUILDDIR)/mkTbAccelerator4x4.bexe
	$<

test-xlu: $(BUILDDIR)/mkTbXLU.bexe
	$<

test-vmem: $(BUILDDIR)/mkTbVMEM.bexe
	$<

test-vregfile: $(BUILDDIR)/mkTbVRegFile.bexe
	$<

test-vpu: $(BUILDDIR)/mkTbVPU.bexe
	$<

test-sxu: $(BUILDDIR)/mkTbScalarUnit.bexe
	$<

test-tc: $(BUILDDIR)/mkTbTensorCore.bexe
	$<

test-sc: $(BUILDDIR)/mkTbSparseCore.bexe
	$<

test: test-pe test-array test-accel test-4x4 test-xlu test-vmem test-vregfile test-vpu test-sxu test-tc test-sc

# --- Dependencies ---

$(BUILDDIR)/TbPE.bo: $(BUILDDIR)/PE.bo
$(BUILDDIR)/SystolicArray.bo: $(BUILDDIR)/PE.bo
$(BUILDDIR)/TbSystolicArray.bo: $(BUILDDIR)/SystolicArray.bo
$(BUILDDIR)/Controller.bo: $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo
$(BUILDDIR)/TensorAccelerator.bo: $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/Controller.bo
$(BUILDDIR)/TbAccelerator.bo: $(BUILDDIR)/TensorAccelerator.bo
$(BUILDDIR)/TbAccelerator4x4.bo: $(BUILDDIR)/TensorAccelerator.bo
$(BUILDDIR)/TbXLU.bo: $(BUILDDIR)/XLU.bo
$(BUILDDIR)/TbVMEM.bo: $(BUILDDIR)/VMEM.bo
$(BUILDDIR)/TbVRegFile.bo: $(BUILDDIR)/VRegFile.bo
$(BUILDDIR)/TbVPU.bo: $(BUILDDIR)/VPU.bo
$(BUILDDIR)/ScalarUnit.bo: $(BUILDDIR)/VMEM.bo $(BUILDDIR)/VRegFile.bo $(BUILDDIR)/VPU.bo $(BUILDDIR)/XLU.bo $(BUILDDIR)/Controller.bo
$(BUILDDIR)/TbScalarUnit.bo: $(BUILDDIR)/ScalarUnit.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo
$(BUILDDIR)/TensorCore.bo: $(BUILDDIR)/ScalarUnit.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/VMEM.bo $(BUILDDIR)/VRegFile.bo $(BUILDDIR)/VPU.bo $(BUILDDIR)/XLU.bo $(BUILDDIR)/Controller.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo
$(BUILDDIR)/TbTensorCore.bo: $(BUILDDIR)/TensorCore.bo
$(BUILDDIR)/TbSparseCore.bo: $(BUILDDIR)/SparseCore.bo

.PHONY: clean test test-pe test-array test-accel test-4x4 test-xlu test-vmem test-vregfile test-vpu test-sxu test-tc test-sc
clean:
	rm -rf $(BUILDDIR)
