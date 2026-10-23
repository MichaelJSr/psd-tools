#target photoshop
// Photoshop as a renderer: open each case, save a PSD copy (with its composite) and a PNG copy, close it.
// make_cases.py fills __DIR__ and __FILES__. Only documents this script opened are closed.
app.displayDialogs = DialogModes.NO;
var DIR = __DIR__, FILES = __FILES__, LOG = [];
function log(s) { LOG.push(s); var f = new File(DIR + "out/render.log"); f.open("w"); f.write(LOG.join("\n")); f.close(); }
new Folder(DIR + "out").create();
var keepCompat = app.preferences.maximizeCompatibility;
app.preferences.maximizeCompatibility = QueryStateType.ALWAYS;
for (var i = 0; i < FILES.length; i++) {
  var doc = null, t0 = new Date().getTime();
  try {
    doc = app.open(new File(DIR + "in/" + FILES[i]));
    var opts = new PhotoshopSaveOptions(); opts.layers = true; opts.embedColorProfile = true;
    doc.saveAs(new File(DIR + "out/" + FILES[i]), opts, true, Extension.LOWERCASE);
    doc.saveAs(new File(DIR + "out/" + FILES[i].replace(/\.psd$/, ".png")), new PNGSaveOptions(), true, Extension.LOWERCASE);
    log("OK " + FILES[i] + " " + (new Date().getTime() - t0) + "ms");
  } catch (e) {
    log("ERR " + FILES[i] + " " + e);
  }
  if (doc !== null) { try { doc.close(SaveOptions.DONOTSAVECHANGES); } catch (e2) {} }
}
app.preferences.maximizeCompatibility = keepCompat;
log("DONE " + FILES.length);
