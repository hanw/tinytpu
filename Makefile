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

test: test-pe test-array test-accel test-4x4 test-xlu test-vmem test-vregfile

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

.PHONY: clean test test-pe test-array test-accel test-4x4 test-xlu test-vmem test-vregfile
clean:
	rm -rf $(BUILDDIR)
