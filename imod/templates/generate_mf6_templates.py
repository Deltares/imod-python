"""
Generate the Jinja2 templates that we use to write the Modflow 6 block files, from the USGS DFN (definition) files.

Makes use of the mf6ivar.py script that is available here:
https://github.com/MODFLOW-USGS/modflow6/blob/d610e4/doc/mf6io/mf6ivar/mf6ivar.py

TODO blocks not yet converted properly: timing, models, exchanges, solutiongroup, perioddata,
    table, continuous, time, timeseries, exchangedata, vertices, cell2d and probably more
TODO also generate docstrings from the descriptions, see mf6ivar.write_desc

Notes from reading mf6io:

For all stress packages we use list based binary input (method urword),
also for RCH and EVT which also support READARRAY with the READASARRAYS option.

begin period 1
    open/close riv/riv.bin (binary)
end period

{% for key, value in dict.items() %}
begin period {{key}}
    open/close {{value}} (binary)
end period
{% endfor %}

For all that uses the READARRAY method, we support the following subset of READARRAY:
LAYERED CONSTANT, CONSTANT, open/close file.bin (binary). This is used for instance in DIS, IC, NPF

begin griddata
    delr LAYERED
        CONSTANT 1
        CONSTANT 2
        CONSTANT 3
end griddata

begin griddata
    delr
        CONSTANT 1
end griddata

begin griddata
    delr
        open/close dis/top.bin (binary)
end griddata

Time series are not supported. Since for list input we only support the binary form,
boundname is not supported.

Lists for the stress packages (CHD, WEL, DRN, RIV, GHB, RCH, and EVT) have an additional BINARY option.
The BINARY option is not supported for the advanced stress packages (LAK, MAW, SFR, UZF).
"""

import importlib.util
import os
import pathlib
import textwrap


def griddata(v):
    name = v["name"]
    if v["name"] in ("delr", "delc"):
        s = f"  {name}\n    {{{{name}}}}"
    else:
        s = f"  {name}\n"
    return s


def block_entry(varname, block, vardict):
    v = vardict[(varname, block)]

    if v.get("tagged") == "false":
        s = "\n"
    else:
        s = f"{varname}\n"

    if block == "period":
        print(f"period block; varname = {varname}")

    elif block == "griddata":
        s = griddata(v)

    # record or recarray
    elif v["type"].startswith("rec"):
        varnames = v["type"].strip().split()[1:]
        s = ""
        for vn in varnames:
            blockentry = block_entry(vn, block, vardict)
            s += f"{blockentry.strip()} "
        if v["type"].startswith("recarray"):
            s = s.strip()
            s = f"{s}\n  {s}\n  {'...'}\n"

    # layered
    elif v["reader"] == "readarray":
        shape = v["shape"]
        if v.get("layered") == "true":
            # if layered is supported according to dfn,
            # and we get passed {layered: True}, add layered keyword
            layered = f" {{% if layered %}}{varname}_layered{{% endif %}}"
        else:
            layered = ""
        s = f"{s}{layered}\n    {{{varname}}}\n"

    # keyword
    elif v["type"] != "keyword":
        vtmp = varname
        if "shape" in v:
            shape = v["shape"]
            vtmp += shape
        s = f"{s} {{{{{vtmp}}}}}\n"

    # if optional, wrap string in square brackets
    if v.get("optional") == "true":
        # TODO if first entry in block, do not slurp whats in front of if
        # TODO if last enty in block, slurp anfer last endif
        s = f"{{%- if {varname} is defined %}}  {s.strip()}\n{{% endif %}}\n"
    else:
        # prepend with indent and return string
        s = f"  {s}\n"

    return s


def write_block(vardict, block):
    s = f"begin {block}\n"
    for (name, b), v in vardict.items():
        if b == block:
            addv = True
            if v.get("in_record") == "true":
                # do not separately include this variable
                # because it is part of a record
                addv = False
            if v.get("block_variable") == "true":
                # do not separately include this variable
                # because it is part of a record
                addv = False
            if addv and (b == "period"):
                print(b, v)
                s = textwrap.dedent(
                    """\
                {% for i, path in periods.items() %}begin period {{i}}
                  open/close {{path}} (binary)
                end period
                {% endfor %}"""
                )
                return s
            if addv:
                ts = block_entry(name, block, vardict)
                s += f"{ts}"
    s += f"end {block}"
    return s


if __name__ == "__main__":

    # path to mf6ivar directory and output directory
    mf6ivar_dir = pathlib.Path("d:/repo/imod/modflow6/doc/mf6io/mf6ivar")
    j2dir = pathlib.Path(__file__).resolve().parent / "mf6"
    os.chdir(mf6ivar_dir)  # is needed for mf6ivar relative paths

    # the below does import mf6ivar, but based on its path
    spec = importlib.util.spec_from_file_location("mf6ivar", mf6ivar_dir / "mf6ivar.py")
    mf6ivar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf6ivar)

    dfndir = mf6ivar_dir / "dfn"
    assert dfndir.is_dir()
    j2dir.mkdir(parents=True, exist_ok=True)

    # construct list of dfn paths to process
    # include the dash to leave out common.dfn
    # dfnpaths = list(dfndir.glob("*-*.dfn"))
    # assert len(dfnpaths) > 30
    dfnpaths = list(dfndir.glob("gwf-dis.dfn"))

    for dfnpath in dfnpaths:
        component, package = dfnpath.stem.split("-", maxsplit=1)
        vardict = mf6ivar.parse_mf6var_file(dfnpath)

        # make list of unique block names
        blocks = []
        for k in vardict:
            v = vardict[k]
            b = v["block"]
            if b not in blocks:
                blocks.append(b)

        # go through each block and write information
        fname = j2dir / (dfnpath.stem + ".j2")
        with open(fname, "w") as f:
            for b in blocks:
                s = write_block(vardict, b) + "\n\n"
                f.write(s)
