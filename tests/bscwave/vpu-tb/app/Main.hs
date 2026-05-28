{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
module Main where

import Data.Bits (shiftR, shiftL, (.&.))
import System.Exit (exitFailure)

import Bscwave.Interface
import Bscwave.Sim
import qualified Bscwave.Waveform as Waveform
import qualified Bscwave.Render as Render
import qualified MkVPU_S1L4 as V

-- VpuOp encoding follows the source order of the enum declaration in
-- src/VPU.bsv (bsc derives Bits in that order). VPU_ADD is first, so 0.
vpuAdd :: Bit 7
vpuAdd = 0

-- bsc's pack for Vector#(n, T) places element 0 in the high bits. For
-- Vector#(1, Vector#(4, Int#(32))) we get a 128-bit value where lane 0
-- occupies bits[127:96] and lane 3 occupies bits[31:0].
packLanes :: [Integer] -> Bit 128
packLanes xs = bit @128 $ foldr (\(i, v) acc -> acc + (mask32 v `shiftL` (96 - 32*i))) 0 (zip [0..] xs)
  where
    mask32 v = v .&. 0xffffffff

unpackLanes :: Bit 128 -> [Integer]
unpackLanes b =
  [ signed32 ((unBit b `shiftR` (96 - 32*i)) .&. 0xffffffff) | i <- [0 .. 3 :: Int] ]
  where
    signed32 v = if v >= 0x80000000 then v - 0x100000000 else v

-- Render a single 32-bit lane as signed decimal.
laneI32 :: Integer -> String
laneI32 v
  | v >= 0x80000000 = show (v - 0x100000000)
  | otherwise       = show v

testbench :: IO (Waveform.Waveform, Integer)
testbench = do
  sim <- create @V.Inputs @V.Outputs V.modelName
  let i = inputs sim
      o = outputs sim

  -- Attach per-lane decimal formatters BEFORE Waveform.create, since the
  -- waveform snapshots the formatter at capture time.
  formatPort (V.execute_src1 i) (lanesFormat 4 32 laneI32)
  formatPort (V.execute_src2 i) (lanesFormat 4 32 laneI32)
  formatPort (V.resultreg    o) (lanesFormat 4 32 laneI32)
  -- The 7-bit op encoding is more readable as decimal than hex here.
  formatPort (V.execute_op   i) decimalFormat

  (waves, sim') <- Waveform.create sim
  let step en op s1 s2 = do
        writePort (V.en_execute   (inputs sim')) en
        writePort (V.execute_op   (inputs sim')) op
        writePort (V.execute_src1 (inputs sim')) (packLanes s1)
        writePort (V.execute_src2 (inputs sim')) (packLanes s2)
        simStep sim'

  -- Dispatch VPU_ADD([1,2,3,4], [10,20,30,40]).
  step 1 vpuAdd [1, 2, 3, 4] [10, 20, 30, 40]
  -- Two idle cycles let resultReg settle and give the wave room to render.
  step 0 0 [0,0,0,0] [0,0,0,0]
  step 0 0 [0,0,0,0] [0,0,0,0]

  out <- readPort (V.resultreg o)
  pure (waves, unBit out)

main :: IO ()
main = do
  (waves, got) <- testbench
  Render.print waves
  let lanes    = unpackLanes (bit @128 got)
      expected = [11, 22, 33, 44]
  putStrLn $ "expected: " <> show expected
  putStrLn $ "got:      " <> show lanes
  if lanes == expected
    then putStrLn "PASS"
    else putStrLn "FAIL" >> exitFailure
