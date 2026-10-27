"""Bokeh visualiser for the (ln alpha, ln beta) NC/CC classifier of classify_gamma.py.

Loads the latest checkpoint of the newest run under logs/classify_gamma/<source>/lightning_logs and shows
P(NC | ln alpha, ln beta) (output scalar 1; 0 is the logit phi, 2 is P(CC)) over the two inputs.

Run from this directory (server options such as --port go before --args, script arguments after it):
    bokeh serve --show vis.py                                   # newest run of either source
    bokeh serve --show --port 5007 vis.py --args pred           # newest run trained on the inferred profile
    bokeh serve --show vis.py --args logs/classify_gamma/truth/lightning_logs/version_0
    bokeh serve --show vis.py --args pred_gev_evisz_s85_both_none_nomatch   # newest run of a tagged build (its directory under logs/classify_gamma)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:                      # bokeh serve runs the script from its own process
    sys.path.insert(0, str(HERE))

from bokeh.io import curdoc                        # noqa: E402
from iwpc.visualise.bokeh_function_visualiser_2D import BokehFunctionVisualiser2D  # noqa: E402

from classify_gamma import LOG_DIR, GammaClassifier, best_checkpoint  # noqa: E402

script_args = [a for a in sys.argv[1:] if not a.startswith("--")]   # bokeh options such as --port belong before --args
arg = script_args[0] if script_args else None
if arg and (LOG_DIR / arg / "lightning_logs").is_dir():               # "truth", "pred" or a tagged build such as pred_<tag>_nomatch
    runs = list((LOG_DIR / arg / "lightning_logs").glob("version_*"))
elif arg:
    runs = [Path(arg)]
else:
    runs = list(LOG_DIR.glob("*/lightning_logs/version_*"))
if not runs:
    raise SystemExit(f"no runs under {LOG_DIR}: train with classify_gamma.py first")
version = max(runs, key=lambda p: p.stat().st_mtime)
ckpt = best_checkpoint(version)                    # best-epoch=K.ckpt (lowest validation loss) of that run
print(f"visualising {ckpt.relative_to(HERE)}")

module = GammaClassifier.load_from_checkpoint(ckpt, map_location="cpu")

curdoc().add_root(BokehFunctionVisualiser2D.visualise(
    module,
    initial_x_axis_scalar_ind=0,          # ln alpha
    initial_y_axis_scalar_ind=1,          # ln beta
    initial_output_scalar_ind=1,          # P(NC | x)
    selected_input_parameter_resolution=600).root)
