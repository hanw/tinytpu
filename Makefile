BSC      = bsc +RTS -K256M -RTS -no-show-timestamps -no-show-version
BUILDDIR = build
TRACEBUILDDIR = $(BUILDDIR)/trace
BSCFLAGS = -sim -p src:test:+ -bdir $(BUILDDIR) -simdir $(BUILDDIR)
TRACE_RUNTIME_SRCS = $(wildcard src/*.bsv) test/TbTinyTPURuntime.bsv bdpi/tinytpu_io.c

$(BUILDDIR):
	mkdir -p $@

$(TRACEBUILDDIR):
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

$(BUILDDIR)/mkTbFpReducer.bexe: $(BUILDDIR)/TbFpReducer.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbFpReducer $(BUILDDIR)/mkTbFpReducer.ba

$(BUILDDIR)/mkTbTranscUnit.bexe: $(BUILDDIR)/TbTranscUnit.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbTranscUnit $(BUILDDIR)/mkTbTranscUnit.ba

$(BUILDDIR)/mkTbPSUMBank.bexe: $(BUILDDIR)/TbPSUMBank.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbPSUMBank $(BUILDDIR)/mkTbPSUMBank.ba

$(BUILDDIR)/mkTbWeightSRAMDB.bexe: $(BUILDDIR)/TbWeightSRAMDB.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbWeightSRAMDB $(BUILDDIR)/mkTbWeightSRAMDB.ba

$(BUILDDIR)/mkTbActivationSRAMDB.bexe: $(BUILDDIR)/TbActivationSRAMDB.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbActivationSRAMDB $(BUILDDIR)/mkTbActivationSRAMDB.ba

$(BUILDDIR)/mkTbScalarUnit.bexe: $(BUILDDIR)/TbScalarUnit.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbScalarUnit $(BUILDDIR)/mkTbScalarUnit.ba

$(BUILDDIR)/mkTbSxuPSUM.bexe: $(BUILDDIR)/TbSxuPSUM.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbSxuPSUM $(BUILDDIR)/mkTbSxuPSUM.ba

$(BUILDDIR)/mkTbCtrlPSUM.bexe: $(BUILDDIR)/TbCtrlPSUM.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbCtrlPSUM $(BUILDDIR)/mkTbCtrlPSUM.ba

$(BUILDDIR)/mkTbCtrlOS.bexe: $(BUILDDIR)/TbCtrlOS.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbCtrlOS $(BUILDDIR)/mkTbCtrlOS.ba

$(BUILDDIR)/mkTbCtrlDB.bexe: $(BUILDDIR)/TbCtrlDB.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbCtrlDB $(BUILDDIR)/mkTbCtrlDB.ba

$(BUILDDIR)/mkTbWeightDMA.bexe: $(BUILDDIR)/TbWeightDMA.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbWeightDMA $(BUILDDIR)/mkTbWeightDMA.ba

$(BUILDDIR)/mkTbActivationDMA.bexe: $(BUILDDIR)/TbActivationDMA.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbActivationDMA $(BUILDDIR)/mkTbActivationDMA.ba

$(BUILDDIR)/mkTbCtrlDBDMA.bexe: $(BUILDDIR)/TbCtrlDBDMA.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbCtrlDBDMA $(BUILDDIR)/mkTbCtrlDBDMA.ba

$(BUILDDIR)/mkTbTensorCore.bexe: $(BUILDDIR)/TbTensorCore.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbTensorCore $(BUILDDIR)/mkTbTensorCore.ba

$(BUILDDIR)/mkTbSparseCore.bexe: $(BUILDDIR)/TbSparseCore.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbSparseCore $(BUILDDIR)/mkTbSparseCore.ba

$(BUILDDIR)/mkTbHBMModel.bexe: $(BUILDDIR)/TbHBMModel.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbHBMModel $(BUILDDIR)/mkTbHBMModel.ba

$(BUILDDIR)/mkTbChipNoC.bexe: $(BUILDDIR)/TbChipNoC.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbChipNoC $(BUILDDIR)/mkTbChipNoC.ba

$(BUILDDIR)/mkTbTinyTPUChip.bexe: $(BUILDDIR)/TbTinyTPUChip.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbTinyTPUChip $(BUILDDIR)/mkTbTinyTPUChip.ba

$(BUILDDIR)/mkTbTinyTPURuntime.bexe: $(BUILDDIR)/TbTinyTPURuntime.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbTinyTPURuntime $(BUILDDIR)/mkTbTinyTPURuntime.ba bdpi/tinytpu_io.c

$(BUILDDIR)/mkTbTinyTPURuntimeTrace.bexe: $(TRACE_RUNTIME_SRCS) | $(BUILDDIR) $(TRACEBUILDDIR)
	$(BSC) -sim -D TRACE -p src:test:+ -bdir $(TRACEBUILDDIR) -simdir $(TRACEBUILDDIR) -u -g mkTbTinyTPURuntime test/TbTinyTPURuntime.bsv
	$(BSC) -sim -D TRACE -p src:test:+ -bdir $(TRACEBUILDDIR) -simdir $(TRACEBUILDDIR) -o $@ -e mkTbTinyTPURuntime $(TRACEBUILDDIR)/mkTbTinyTPURuntime.ba bdpi/tinytpu_io.c

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

test-fpreducer: $(BUILDDIR)/mkTbFpReducer.bexe
	$<

test-transcunit: $(BUILDDIR)/mkTbTranscUnit.bexe
	$<

test-psumbank: $(BUILDDIR)/mkTbPSUMBank.bexe
	$<

test-wsram-db: $(BUILDDIR)/mkTbWeightSRAMDB.bexe
	$<

test-asram-db: $(BUILDDIR)/mkTbActivationSRAMDB.bexe
	$<

test-sxu: $(BUILDDIR)/mkTbScalarUnit.bexe
	$<

test-sxu-psum: $(BUILDDIR)/mkTbSxuPSUM.bexe
	$<

test-ctrl-psum: $(BUILDDIR)/mkTbCtrlPSUM.bexe
	$<

test-ctrl-os: $(BUILDDIR)/mkTbCtrlOS.bexe
	$<

test-ctrl-db: $(BUILDDIR)/mkTbCtrlDB.bexe
	$<

test-weight-dma: $(BUILDDIR)/mkTbWeightDMA.bexe
	$<

test-activ-dma: $(BUILDDIR)/mkTbActivationDMA.bexe
	$<

test-ctrl-db-dma: $(BUILDDIR)/mkTbCtrlDBDMA.bexe
	$<

test-tc: $(BUILDDIR)/mkTbTensorCore.bexe
	$<

test-sc: $(BUILDDIR)/mkTbSparseCore.bexe
	$<

test-hbm: $(BUILDDIR)/mkTbHBMModel.bexe
	$<

test-noc: $(BUILDDIR)/mkTbChipNoC.bexe
	$<

test-chip: $(BUILDDIR)/mkTbTinyTPUChip.bexe
	$<

runtime-tb: $(BUILDDIR)/mkTbTinyTPURuntime.bexe

runtime-tb-trace: $(BUILDDIR)/mkTbTinyTPURuntimeTrace.bexe

test-trace: runtime-tb-trace
	python3 scripts/profiler/sample_program.py /tmp/tinytpu_profile_sample.txt
	TINYTPU_BUNDLE=/tmp/tinytpu_profile_sample.txt ./build/mkTbTinyTPURuntimeTrace.bexe > /tmp/tinytpu_profile_trace.out
	grep -q "TRACE cycle=" /tmp/tinytpu_profile_trace.out
	grep -q "status ok" /tmp/tinytpu_profile_trace.out

test: test-pe test-array test-accel test-4x4 test-xlu test-vmem test-vregfile test-vpu test-sxu test-tc test-sc test-hbm test-noc test-chip

# --- Dependencies ---

$(BUILDDIR)/TbPE.bo: $(BUILDDIR)/PE.bo
$(BUILDDIR)/SystolicArray.bo: $(BUILDDIR)/PE.bo
$(BUILDDIR)/TbSystolicArray.bo: $(BUILDDIR)/SystolicArray.bo
$(BUILDDIR)/Controller.bo: $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/TensorAccelerator.bo: $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/Controller.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/TbAccelerator.bo: $(BUILDDIR)/TensorAccelerator.bo
$(BUILDDIR)/TbAccelerator4x4.bo: $(BUILDDIR)/TensorAccelerator.bo
$(BUILDDIR)/TbXLU.bo: $(BUILDDIR)/XLU.bo
$(BUILDDIR)/TbVMEM.bo: $(BUILDDIR)/VMEM.bo
$(BUILDDIR)/TbVRegFile.bo: $(BUILDDIR)/VRegFile.bo
$(BUILDDIR)/VPU.bo: $(BUILDDIR)/FpReducer.bo $(BUILDDIR)/TranscUnit.bo
$(BUILDDIR)/TbVPU.bo: $(BUILDDIR)/VPU.bo
$(BUILDDIR)/TbFpReducer.bo: $(BUILDDIR)/FpReducer.bo
$(BUILDDIR)/TbTranscUnit.bo: $(BUILDDIR)/TranscUnit.bo
$(BUILDDIR)/TbPSUMBank.bo: $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/WeightSRAMDB.bo: $(BUILDDIR)/WeightSRAM.bo
$(BUILDDIR)/ActivationSRAMDB.bo: $(BUILDDIR)/ActivationSRAM.bo
$(BUILDDIR)/TbWeightSRAMDB.bo: $(BUILDDIR)/WeightSRAMDB.bo
$(BUILDDIR)/TbActivationSRAMDB.bo: $(BUILDDIR)/ActivationSRAMDB.bo
$(BUILDDIR)/ScalarUnit.bo: $(BUILDDIR)/VMEM.bo $(BUILDDIR)/VRegFile.bo $(BUILDDIR)/VPU.bo $(BUILDDIR)/XLU.bo $(BUILDDIR)/Controller.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/TbScalarUnit.bo: $(BUILDDIR)/ScalarUnit.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/TbSxuPSUM.bo: $(BUILDDIR)/ScalarUnit.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/TbCtrlPSUM.bo: $(BUILDDIR)/Controller.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/TbCtrlOS.bo: $(BUILDDIR)/Controller.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/TbCtrlDB.bo: $(BUILDDIR)/Controller.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/WeightSRAMDB.bo $(BUILDDIR)/ActivationSRAMDB.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/WeightDMA.bo: $(BUILDDIR)/WeightSRAMDB.bo
$(BUILDDIR)/TbWeightDMA.bo: $(BUILDDIR)/WeightDMA.bo $(BUILDDIR)/WeightSRAMDB.bo
$(BUILDDIR)/ActivationDMA.bo: $(BUILDDIR)/ActivationSRAMDB.bo
$(BUILDDIR)/TbActivationDMA.bo: $(BUILDDIR)/ActivationDMA.bo $(BUILDDIR)/ActivationSRAMDB.bo
$(BUILDDIR)/TbCtrlDBDMA.bo: $(BUILDDIR)/Controller.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/WeightSRAMDB.bo $(BUILDDIR)/WeightDMA.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/TensorCore.bo: $(BUILDDIR)/ScalarUnit.bo $(BUILDDIR)/SystolicArray.bo $(BUILDDIR)/VMEM.bo $(BUILDDIR)/VRegFile.bo $(BUILDDIR)/VPU.bo $(BUILDDIR)/XLU.bo $(BUILDDIR)/Controller.bo $(BUILDDIR)/WeightSRAM.bo $(BUILDDIR)/ActivationSRAM.bo $(BUILDDIR)/PSUMBank.bo
$(BUILDDIR)/TbTensorCore.bo: $(BUILDDIR)/TensorCore.bo
$(BUILDDIR)/TbSparseCore.bo: $(BUILDDIR)/SparseCore.bo
$(BUILDDIR)/TbHBMModel.bo: $(BUILDDIR)/HBMModel.bo
$(BUILDDIR)/TbChipNoC.bo: $(BUILDDIR)/ChipNoC.bo
$(BUILDDIR)/TinyTPUChip.bo: $(BUILDDIR)/TensorCore.bo $(BUILDDIR)/SparseCore.bo $(BUILDDIR)/HBMModel.bo $(BUILDDIR)/ChipNoC.bo
$(BUILDDIR)/TbTinyTPUChip.bo: $(BUILDDIR)/TinyTPUChip.bo
$(BUILDDIR)/TbTinyTPURuntime.bo: $(BUILDDIR)/TensorCore.bo

.PHONY: clean test test-pe test-array test-accel test-4x4 test-xlu test-vmem test-vregfile test-vpu test-fpreducer test-psumbank test-sxu test-sxu-psum test-ctrl-psum test-ctrl-os test-tc test-sc test-hbm test-noc test-chip runtime-tb runtime-tb-trace test-trace
clean:
	rm -rf $(BUILDDIR)
