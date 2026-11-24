//! VapourSynth plugin entry point for `zexpr`.

const vapoursynth = @import("vapoursynth");
const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;
const zon = @import("zon");
const expr = @import("vs/expr.zig");
const select = @import("vs/select.zig");
const prop_expr = @import("vs/prop_expr.zig");

fn pluginTeardown() callconv(.c) void {
    expr.shutdownCache();
}

export fn VapourSynthPluginInit2(plugin: *vs.Plugin, vspapi: *const vs.PLUGINAPI) void {
    expr.registerTeardownHook(&pluginTeardown);
    ZAPI.Plugin.config("com.julek.zexpr", "zexpr", "AsmJit-accelerated Expr plugin", zon.version, plugin, vspapi);
    ZAPI.Plugin.function(expr.filter_name, expr.params, "clip:vnode;", expr.create, plugin, vspapi);
    ZAPI.Plugin.function(select.filter_name, select.params, "clip:vnode;", select.create, plugin, vspapi);
    ZAPI.Plugin.function(prop_expr.filter_name, prop_expr.params, "clip:vnode;", prop_expr.create, plugin, vspapi);
}
