import abc
from dataclasses import dataclass
from typing import List, Optional, Union

from imod import declxml as xml


class Aggregate(abc.ABC):
    pass


class Attribute(abc.ABC):
    pass


# Mapcanvas
@dataclass
class SpatialRefSys(Aggregate):
    # proj4: str = ""
    wkt: Optional[str] = None
    # srsid: int = 0
    # srid: int = 0
    # authid: str = ""
    # description: str = ""
    # projectionacronym: str = ""
    # ellipsoidacronym: str = ""
    geographicflag: bool = False


@dataclass
class DestinationSrs(Aggregate):
    spatialrefsys: SpatialRefSys = SpatialRefSys()


@dataclass
class ProjectCrs(Aggregate):
    spatialrefsys: SpatialRefSys = SpatialRefSys()


@dataclass
class Extent(Aggregate):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class MapCanvas(Aggregate):
    extent: Extent
    annotationsVisible: Union[Attribute, str] = "1"
    name: Union[Attribute, str] = "theMapCanvas"
    units: str = "unknown"
    rotation: float = 0.0
    rendermaptile: int = 0
    expressionContextScope: str = ""
    destinationsrs: Optional[DestinationSrs] = None


# Layer tree
@dataclass
class Layer(Aggregate):
    id: Union[Attribute, str]


@dataclass
class Layer_Tree_Layer(Aggregate):
    id: Union[Attribute, str]
    name: Union[Attribute, str]
    source: Union[Attribute, str]
    providerKey: Union[Attribute, str] = "mdal"
    expanded: Union[Attribute, str] = "1"
    checked: Union[Attribute, str] = "Qt::Checked"
    customproperties: str = ""


@dataclass
class Custom_Order(Aggregate):
    item: List[str]
    enabled: Union[Attribute, str] = "0"


@dataclass
class Layer_Tree_Group_Leaf(Aggregate):
    layer_tree_layer: List[Layer_Tree_Layer]
    custom_order: Custom_Order
    customproperties: str = ""
    checked: Optional[Union[Attribute, str]] = None
    name: Optional[Union[Attribute, str]] = None
    expanded: Optional[Union[Attribute, str]] = None


@dataclass
class Layer_Tree_Group_Root(Aggregate):
    layer_tree_group_leaf: List[Layer_Tree_Group_Leaf]
    customproperties: str = ""
    custom_order: Optional[Custom_Order] = None
    checked: Optional[Union[Attribute, str]] = None
    name: Optional[Union[Attribute, str]] = None
    expanded: Optional[Union[Attribute, str]] = None


# Snapping
@dataclass
class Snapping_Settings(Aggregate):
    type: Union[Attribute, str] = "1"
    intersection_snapping: Union[Attribute, str] = "0"
    unit: Union[Attribute, str] = "1"
    enabled: Union[Attribute, str] = "0"
    tolerance: Union[Attribute, str] = "12"
    mode: Union[Attribute, str] = "2"
    individual_layer_settings: str = ""


# Mesh renderer and settings
@dataclass
class Prop(Aggregate):
    k: Union[Attribute, str]
    v: Union[Attribute, str]


default_colorramp = (  # lists not accepted as default because they are mutable, so has to be tuple
    Prop("color1", "13,8,135,255"),
    Prop("color2", "240,249,33,255"),
    Prop("discrete", "0"),
    Prop("rampType", "gradient"),
    # TODO: Write color gradient creator to generate these strings
    # (get them from matplotlib or something)
    Prop(
        "stops",
        "0.0196078;27,6,141,255:0.0392157;38,5,145,255:0.0588235;47,5,150,255:0.0784314;56,4,154,255:0.0980392;65,4,157,255:0.117647;73,3,160,255:0.137255;81,2,163,255:0.156863;89,1,165,255:0.176471;97,0,167,255:0.196078;105,0,168,255:0.215686;113,0,168,255:0.235294;120,1,168,255:0.254902;128,4,168,255:0.27451;135,7,166,255:0.294118;142,12,164,255:0.313725;149,17,161,255:0.333333;156,23,158,255:0.352941;162,29,154,255:0.372549;168,34,150,255:0.392157;174,40,146,255:0.411765;180,46,141,255:0.431373;186,51,136,255:0.45098;191,57,132,255:0.470588;196,62,127,255:0.490196;201,68,122,255:0.509804;205,74,118,255:0.529412;210,79,113,255:0.54902;214,85,109,255:0.568627;218,91,105,255:0.588235;222,97,100,255:0.607843;226,102,96,255:0.627451;230,108,92,255:0.647059;233,114,87,255:0.666667;237,121,83,255:0.686275;240,127,79,255:0.705882;243,133,75,255:0.72549;245,140,70,255:0.745098;247,147,66,255:0.764706;249,154,62,255:0.784314;251,161,57,255:0.803922;252,168,53,255:0.823529;253,175,49,255:0.843137;254,183,45,255:0.862745;254,190,42,255:0.882353;253,198,39,255:0.901961;252,206,37,255:0.921569;251,215,36,255:0.941176;248,223,37,255:0.960784;246,232,38,255:0.980392;243,240,39,255",
    ),
)


@dataclass
class Mesh_Settings_Native(Aggregate):
    color: Union[Attribute, str] = "0,0,0,255"
    enabled: Union[Attribute, str] = "0"
    line_width: Union[Attribute, str] = "0.26"
    line_width_unit: Union[Attribute, str] = "MM"


@dataclass
class Mesh_Settings_Triangular(Aggregate):
    color: Union[Attribute, str] = "0,0,0,255"
    enabled: Union[Attribute, str] = "0"
    line_width: Union[Attribute, str] = "0.26"
    line_width_unit: Union[Attribute, str] = "MM"


@dataclass
class Mesh_Settings_Edge(Aggregate):
    color: Union[Attribute, str] = "0,0,0,255"
    enabled: Union[Attribute, str] = "0"
    line_width: Union[Attribute, str] = "0.26"
    line_width_unit: Union[Attribute, str] = "MM"


@dataclass
class Item(Aggregate):
    label: Union[Attribute, str]
    value: Union[Attribute, str]
    color: Union[Attribute, str]
    alpha: Union[Attribute, str] = "255"


@dataclass
class ColorRamp(Aggregate):
    prop: List[Prop] = default_colorramp
    type: Union[Attribute, str] = "gradient"
    name: Union[Attribute, str] = "[source]"


@dataclass
class ColorRampShader(Aggregate):
    colorramp: ColorRamp = ColorRamp()
    item: Optional[List[Item]] = None
    classificationMode: Union[Attribute, str] = "1"
    colorRampType: Union[Attribute, str] = "INTERPOLATED"
    clip: Union[Attribute, str] = "0"


@dataclass
class Mesh_Stroke_Width(Aggregate):
    ignore_out_range: Union[Attribute, str] = "0"
    maximum_width: Union[Attribute, str] = "0"
    width_varying: Union[Attribute, str] = "0"
    minimum_value: Union[Attribute, str] = "0"
    use_absolute_value: Union[Attribute, str] = "0"
    fixed_width: Union[Attribute, str] = "0"
    minimum_width: Union[Attribute, str] = "0"


@dataclass
class Edge_Settings(Aggregate):
    mesh_stroke_width: Mesh_Stroke_Width = Mesh_Stroke_Width()
    stroke_width_unit: Union[Attribute, str] = "0"


@dataclass
class Scalar_Settings(Aggregate):
    group: Union[Attribute, str]
    min_val: Union[Attribute, str]
    max_val: Union[Attribute, str]
    colorrampshader: ColorRampShader = ColorRampShader()
    opacity: Union[Attribute, str] = "1"
    interpolation_method: Union[Attribute, str] = "none"
    edge_settings: Edge_Settings = Edge_Settings()


@dataclass
class Activate_Dataset(Aggregate):
    scalar: Union[Attribute, str] = "0,0"


@dataclass
class Mesh_Renderer_Settings(Aggregate):
    scalar_settings: List[Scalar_Settings]
    activate_dataset: Activate_Dataset = Activate_Dataset()
    mesh_settings_native: Mesh_Settings_Native = Mesh_Settings_Native()
    mesh_settings_edge: Mesh_Settings_Edge = Mesh_Settings_Edge()
    mesh_settings_triangular: Mesh_Settings_Triangular = Mesh_Settings_Triangular()


# Projectlayers
@dataclass
class KeywordList(Aggregate):
    value: str = ""


@dataclass
class NoDataList(Aggregate):
    bandNo: Union[Attribute, str]
    useSrcNoData: Union[Attribute, str]


@dataclass
class NoData(Aggregate):
    nodatalist: NoDataList


@dataclass
class Srs(Aggregate):
    spatialrefsys: SpatialRefSys = SpatialRefSys()


@dataclass
class Map_Layer_Style(Aggregate):
    name: Union[Attribute, str]


@dataclass
class Map_Layer_Style_Manager(Aggregate):
    current: Union[Attribute, str]
    map_layer_style: Map_Layer_Style


@dataclass
class Flags(Aggregate):
    Identifiable: int = 1
    Removable: int = 1
    Searchable: int = 1


@dataclass
class Provider(
    Aggregate
):  # Probably should inherit from Attribute to get what we need?
    provider: str = "mdal"
    time_unit: Optional[Union[Attribute, str]] = None


@dataclass
class Temporal(Aggregate):
    end_time_extent: Union[Attribute, str]
    reference_time: Union[Attribute, str]
    start_time_extent: Union[Attribute, str]
    matching_method: Union[Attribute, str] = "0"
    temporal_active: Union[Attribute, str] = "1"


@dataclass
class Mesh_Dataset_Group_Tree_Item(Aggregate):
    provider_name: Union[Attribute, str]
    display_name: Union[Attribute, str] = ""
    dataset_index: Union[Attribute, str] = "0"
    is_vector: Union[Attribute, str] = "0"
    is_enabled: Union[Attribute, str] = "1"


@dataclass
class Dataset_Groups_Tree(Aggregate):
    mesh_dataset_group_tree_item: List[Mesh_Dataset_Group_Tree_Item]


@dataclass
class Mesh_Simplify_Settings(Aggregate):
    mesh_resolution: Union[Attribute, str] = "5"
    enabled: Union[Attribute, str] = "0"
    reduction_factor: Union[Attribute, str] = "10"


@dataclass
class MapLayer(Aggregate):  # Attributes of this class seem to be required.
    extent: Extent = None
    id: str = None
    datasource: str = None
    keywordlist: KeywordList = KeywordList()
    layername: str = None
    srs: Srs = Srs()
    provider: Provider = Provider()
    dataset_groups_tree: Dataset_Groups_Tree = None
    flags: Flags = Flags()
    temporal: Optional[Temporal] = None
    customproperties: str = ""
    mesh_renderer_settings: Mesh_Renderer_Settings = None
    mesh_simplify_settings: Mesh_Simplify_Settings = Mesh_Simplify_Settings()
    blendMode: int = 0

    hasScaleBasedVisibilityFlag: Union[Attribute, str] = "0"
    styleCategories: Union[Attribute, str] = "AllStyleCategories"
    type: Union[Attribute, str] = "mesh"
    minScale: Union[Attribute, str] = "1e+08"
    maxScale: Union[Attribute, str] = "0"
    autoRefreshEnabled: Union[Attribute, str] = "0"
    autoRefreshTime: Union[Attribute, str] = "0"
    refreshOnNotifyEnabled: Union[Attribute, str] = "0"
    refreshOnNotifyMessage: Union[Attribute, str] = ""


@dataclass
class ProjectLayers(Aggregate):
    maplayer: List[MapLayer]


# Legend
@dataclass
class LegendLayerFile(Aggregate):
    layerid: Union[Attribute, str]
    isInOverview: Union[Attribute, str] = "1"
    visible: Union[Attribute, str] = "0"


@dataclass
class FileGroup(Aggregate):
    legendlayerfile: LegendLayerFile
    hidden: Union[Attribute, str] = "false"
    open: Union[Attribute, str] = "false"


@dataclass
class LegendLayer(Aggregate):
    name: Union[Attribute, str]
    filegroup: FileGroup
    showFeatureCount: Union[Attribute, str] = "0"
    open: Union[Attribute, str] = "false"
    drawingOrder: Union[Attribute, str] = "-1"
    checked: Union[Attribute, str] = "Qt::Checked"


@dataclass
class Legend(Aggregate):
    legendlayer: List[LegendLayer]
    updateDrawingOrder: Union[Attribute, str] = "true"


# properties
@dataclass
class SpatialRefSys_Property(Aggregate):
    ProjectionsEnabled: int = 1
    type: Union[Attribute, str] = "int"


@dataclass
class Properties(Aggregate):
    spatialrefsys_property: SpatialRefSys_Property = SpatialRefSys_Property()


# qgis
@dataclass
class SrcDest(Aggregate):
    src: SpatialRefSys
    dest: SpatialRefSys


@dataclass
class TransformContext(Aggregate):
    srcdest: SrcDest


@dataclass
class HomePath(Aggregate):
    path: Union[Attribute, str] = ""


@dataclass
class AutoTransaction(Aggregate):
    active: Union[Attribute, str] = "0"


@dataclass
class EvaluateDefaultValues(Aggregate):
    active: Union[Attribute, str] = "0"


@dataclass
class Trust(Aggregate):
    active: Union[Attribute, str] = "0"


@dataclass
class Title(Aggregate):
    value: Optional[Trust] = None


@dataclass
class LayerOrder(Aggregate):
    layer: List[Layer]


@dataclass
class Qgis(Aggregate):
    homepath: HomePath = HomePath()
    title: str = "my_project"
    autotransaction: AutoTransaction = AutoTransaction()
    evaluatedefaultvalues: EvaluateDefaultValues = EvaluateDefaultValues()
    trust: Trust = Trust()
    projectcrs: ProjectCrs = ProjectCrs()
    layer_tree_group_leaf: Optional[Layer_Tree_Group_Leaf] = None
    layer_tree_group_root: Optional[Layer_Tree_Group_Root] = None
    snapping_settings: Snapping_Settings = Snapping_Settings()
    relations: str = ""
    mapcanvas: MapCanvas = None
    projectModels: str = ""
    legend: Legend = None
    mapViewDocks: str = ""
    mapViewDocks3D: str = ""
    projectlayers: ProjectLayers = None
    layerorder: LayerOrder = None
    visibility_presets: str = ""
    Annotations: str = ""
    Layouts: str = ""
    Bookmarks: str = ""
    transformcontext: str = ""
    properties: Properties = Properties()
    # transformcontext: Union[TransformContext, str] = ""
    saveUser: Union[Attribute, str] = "imod-python"
    version: Union[Attribute, str] = "3.14.15-Pi"
    saveUserFull: Union[Attribute, str] = "Deltares"
    projectname: Union[Attribute, str] = ""


# Mappings
type_mapping = {
    bool: xml.boolean,
    float: xml.floating_point,
    int: xml.integer,
    str: xml.string,
}

name_mapping = {
    NoData: "noData",
    NoDataList: "noDataList",
    KeywordList: "keywordList",
    ProjectCrs: "projectCrs",
    TransformContext: "transformContext",
    EvaluateDefaultValues: "evaluateDefaultValues",
    HomePath: "homePath",
    Layer_Tree_Group_Leaf: "layer-tree-group",
    Layer_Tree_Group_Root: "layer-tree-group",
    SrcDest: "srcDest",
    SpatialRefSys_Property: "SpatialRefSys",
}

# Functions
# Following dataformats are now supported:
# ("Any" is both Aggregate and Primitive here, where "Primitive" is a placeholder for anything type_mapping)
# -Optional[List[Any]]
# -Optional[Union[Attribute, Primitive]]
# -List[Any]
# -Union[Attribute, Primitive]
# -Optional[Any]
# -Any


def unpack(vartype):
    # List[str] -> [typing.List[str], str]
    # Optional[List[Layer_Tree_Group_Leaf]] -> [Optional[List[Layer_Tree_Group_Leaf]], List[Layer_Tree_Group_Leaf], Layer_Tree_Group_Leaf]
    # ... and so forth
    # An attribute is returned as is:
    # Union[Attribute, str] -> [Union[Attribute, str]]
    # and:
    # List[Union[Attribute, str]] -> [List[Union[Attribute, str], Union[Attribute, str]]
    # i.e. the attribute information is maintained.
    yield vartype
    while hasattr(vartype, "__args__"):
        if is_attribute(vartype):
            return vartype
        vartype = vartype.__args__[0]
        yield vartype


def is_aggregate(vartype):
    try:
        return issubclass(vartype, Aggregate)
    except TypeError:
        return False


def is_required(vartype):
    # Optional is a Union[..., NoneType]
    NoneType = type(None)
    return not (hasattr(vartype, "__args__") and (vartype.__args__[-1] is NoneType))


def is_attribute(vartype):
    try:
        return issubclass(vartype, Attribute)
    except TypeError:
        return hasattr(vartype, "__args__") and (vartype.__args__[0] is Attribute)


def is_list(vartype):
    return hasattr(vartype, "__origin__") and (vartype.__origin__ is list)


def qgis_xml_name(datacls):
    # the qgis xml entries have dashes rather than underscores but dashes aren't
    # valid Python syntax.
    return name_mapping.get(datacls, datacls.__name__.lower().replace("_", "-"))


def process_primitive(name, vartype, datacls, required):
    field_kwargs = {
        "element_name": ".",
        "attribute": name.replace("_", "-"),
        "alias": name,
        "required": required,
        "default": False if required else None,
    }

    if is_attribute(datacls):
        xml_type = type_mapping[vartype]
    elif is_attribute(vartype):
        xml_type = type_mapping[vartype.__args__[1]]
    else:
        xml_type = type_mapping[vartype]
        field_kwargs["element_name"] = field_kwargs.pop("attribute")

    field = xml_type(**field_kwargs)
    return field


def special_case_SpatialRefSys_Property(datacls):
    """
    The current recursive strategy does not support primitives with an attribute
    assigned to them. This is luckily only needed once in our current
    approach, namely to set a property, so we special case this.

    Returns
    -------
    xml.user_object

    """
    children = [
        xml.integer("ProjectionsEnabled"),
        xml.string("ProjectionsEnabled", attribute="type"),
    ]

    return xml.user_object(
        element_name=qgis_xml_name(datacls),
        cls=datacls,
        child_processors=children,
        alias=datacls.__name__.lower(),
        required=True,
    )


def make_processor(datacls: type, element_required: bool = True):
    """
    This is a utility to automate setting up of xml_preprocessors from the
    dataclass annotations. Nested aggregate types are dealt with via recursion.
    """
    if datacls == SpatialRefSys_Property:
        return special_case_SpatialRefSys_Property(datacls)

    children = []
    for name, vartype in datacls.__annotations__.items():
        required = element_required and is_required(vartype)
        type_info = [a for a in unpack(vartype)]
        if len(type_info) > 0:
            vartype = type_info[-1]

        # recursive case: an aggregate type
        if any(is_aggregate(a) for a in type_info):
            child = make_processor(vartype, required)
        # base case: a primitive type
        else:
            child = process_primitive(name, vartype, datacls, required)

        # Deal with arrays
        if any(is_list(a) for a in type_info):
            children.append(xml.array(child))
        else:
            children.append(child)

    return xml.user_object(
        element_name=qgis_xml_name(datacls),
        cls=datacls,
        child_processors=children,
        alias=datacls.__name__.lower(),
        required=element_required,
    )
