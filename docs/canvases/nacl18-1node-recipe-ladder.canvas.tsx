import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Pill,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

/**
 * NaCl 18³ — new 1-node recipe ladder (fxpu_1node_ef_atoms + 10_1node_ef_atoms.pbs).
 * Job 8735819 · pbs/out/1node_ef_ladder_n18/
 */

type Row = {
  W: number;
  E: number;
  Fmax: number;
  load_s: number;
  warmup_s: number;
  ef_mean_s: number;
  wall_s: number;
  dE_meV: number | null;
  maxdF: number | null;
  cos: number | null;
};

const ROWS: Row[] = [
  {
    W: 1,
    E: -157578.53111522,
    Fmax: 0.719101,
    load_s: 45.79,
    warmup_s: 33.56,
    ef_mean_s: 23.255,
    wall_s: 172.65,
    dE_meV: null,
    maxdF: null,
    cos: null,
  },
  {
    W: 2,
    E: -157578.53111522,
    Fmax: 0.719101,
    load_s: 150.06,
    warmup_s: 35.7,
    ef_mean_s: 13.07,
    wall_s: 238.05,
    dE_meV: 3.119e-12,
    maxdF: 1.055e-15,
    cos: 1.0,
  },
  {
    W: 4,
    E: -157578.53111522,
    Fmax: 0.719101,
    load_s: 24.22,
    warmup_s: 9.57,
    ef_mean_s: 6.624,
    wall_s: 60.28,
    dE_meV: 2.495e-12,
    maxdF: 1.166e-15,
    cos: 1.0,
  },
  {
    W: 6,
    E: -157578.53111522,
    Fmax: 0.719101,
    load_s: 21.92,
    warmup_s: 7.54,
    ef_mean_s: 4.489,
    wall_s: 47.42,
    dE_meV: 4.99e-12,
    maxdF: 1.159e-15,
    cos: 1.0,
  },
  {
    W: 12,
    E: -157578.53111522,
    Fmax: 0.719101,
    load_s: 26.51,
    warmup_s: 6.02,
    ef_mean_s: 2.433,
    wall_s: 42.26,
    dE_meV: 4.99e-12,
    maxdF: 1.11e-15,
    cos: 1.0,
  },
];

const EF1 = ROWS[0].ef_mean_s;

function sci(x: number | null, dig = 3): string {
  if (x === null) return "—";
  if (x === 0) return "0";
  return x.toExponential(dig);
}

export default function Nacl18OneNodeRecipeLadder() {
  const parityRows = ROWS.map((r) => [
    String(r.W),
    r.E.toFixed(8),
    r.Fmax.toFixed(6),
    sci(r.dE_meV),
    sci(r.maxdF),
    r.cos === null ? "—" : r.cos.toFixed(10),
    r.cos === null || r.cos >= 0.999999999 ? "PASS" : "FAIL",
  ]);

  const timingRows = ROWS.map((r) => {
    const sp = EF1 / r.ef_mean_s;
    return [
      String(r.W),
      r.load_s.toFixed(2),
      r.warmup_s.toFixed(2),
      r.ef_mean_s.toFixed(3),
      r.wall_s.toFixed(2),
      sp.toFixed(2) + "×",
      ((100 * sp) / r.W).toFixed(0) + "%",
    ];
  });

  const efChart = ROWS.map((r) => ({
    label: `W=${r.W}`,
    value: Number(r.ef_mean_s.toFixed(3)),
  }));
  const speedChart = ROWS.map((r) => ({
    label: `W=${r.W}`,
    value: Number((EF1 / r.ef_mean_s).toFixed(2)),
  }));

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>NaCl 18³ — 1-node recipe E+F ladder</H1>
        <Text tone="secondary" size="small">
          Recipe: pbs/10_1node_ef_atoms.pbs + scripts/fxpu_1node_ef_atoms.py.
          Job 8735819 · out pbs/out/1node_ef_ladder_n18/ · 46 656 atoms · FP64 ·
          XCCL broadcast+sockets · phase1 on · repeats=3 warm ef_mean · vs W=1
          AG forces. Validated 2026-08-05.
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value="5/5" label="E+F vs W=1 PASS" tone="success" />
        <Stat value="9.56×" label="ef_mean W1→W12" tone="success" />
        <Stat value="2.43 s" label="W=12 ef_mean" />
        <Stat value="≤1.2e−15" label="max |ΔF| vs W1" tone="success" />
      </Grid>

      <Callout tone="success" title="Parity">
        Energy identical to −157578.53111522 eV at all W. Force cosine = 1.0;
        max|ΔF| ~1e−15 eV/Å (FP noise). |ΔE|/N ≪ 1e−6 meV/atom.
      </Callout>

      <Card>
        <CardHeader trailing={<Pill active size="sm">parity</Pill>}>
          Energy and AG force parity vs W=1
        </CardHeader>
        <CardBody>
          <Table
            framed
            striped
            headers={[
              "W",
              "E (eV)",
              "Fmax AG",
              "ΔE meV/atom",
              "max|ΔF|",
              "cos F",
              "vs W1",
            ]}
            columnAlign={[
              "right",
              "right",
              "right",
              "right",
              "right",
              "right",
              "left",
            ]}
            rowTone={["success", "success", "success", "success", "success"]}
            rows={parityRows}
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader trailing={<Pill active size="sm">timing</Pill>}>
          Warm E+F timing (load / warmup / ef_mean)
        </CardHeader>
        <CardBody>
          <Table
            framed
            striped
            headers={[
              "W",
              "load_s",
              "warmup_s",
              "ef_mean_s",
              "wall_s",
              "vs W1",
              "eff %",
            ]}
            columnAlign={[
              "right",
              "right",
              "right",
              "right",
              "right",
              "right",
              "right",
            ]}
            rows={timingRows}
          />
          <Text tone="tertiary" size="small">
            ef_mean = mean of 3 timed E+F after warmup (positions nudged to bust
            ASE cache + XPU sync). eff% = 100×(ef_W1/ef_W)/W. W=2 load is cold
            first Ray bring-up.
          </Text>
        </CardBody>
      </Card>

      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Warm ef_mean by W</CardHeader>
          <CardBody>
            <BarChart
              categories={efChart.map((d) => d.label)}
              series={[{ name: "ef_mean_s", data: efChart.map((d) => d.value) }]}
              height={220}
              valueSuffix=" s"
              showValues
            />
            <Text tone="tertiary" size="small">
              Y: seconds · Source: 1node_ef_ladder_n18 · job 8735819
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Speedup vs W=1 (ef_mean)</CardHeader>
          <CardBody>
            <BarChart
              categories={speedChart.map((d) => d.label)}
              series={[
                {
                  name: "ef_W1 / ef_W",
                  data: speedChart.map((d) => d.value),
                },
              ]}
              height={220}
              valueSuffix="×"
              showValues
            />
            <Text tone="tertiary" size="small">
              Ideal linear = W; W=12 reaches 9.56× (~80% efficiency)
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={6}>
        <H2>Artifacts</H2>
        <Text size="small">
          pbs/out/1node_ef_ladder_n18/ladder_summary.json · REPORT.md ·
          forces_wXX.npy · PBS 10_1node_ef_atoms.pbs · scripts/fxpu_1node_ef_atoms.py
        </Text>
      </Stack>
    </Stack>
  );
}
