#!/usr/bin/env python
"""
import weightNode_io

# Find all the weight Nodes
weightDrivers = pm.ls(type="weightDriver")

# any filePath
testPath = r"C:\\Users\rafael\Documents\core\scripts\testWeightNodes.json"

# Export listed weightDrivers
weightNode_io.exportNodes(testPath, weightDrivers)

# import all weight drivers from filePath
weightNode_io.importNodes(testPath)

Attributes:
    CTL_SUFFIX (str): ctl suffix shared from rbf_io
    DRIVEN_SUFFIX (str): suffix shared from rbf_io
    ENVELOPE_ATTR (str): name of the attr that disables rbf node(non enum)
    RBF_TYPE (str): core/plugin node type
    WD_SUFFIX (str): name of the suffix for this rbf node type
    WNODE_DRIVERPOSE_ATTRS (dict): attrs and their type for querying/setting
    WNODE_SHAPE_ATTRS (list): of attrs to query for re-setting on create
    WNODE_TRANSFORM_ATTRS (list): of transform attrs to record

__author__ = "Takayoshi Matsumoto"
__email__ = "yamahigashi@gmail.com"

"""
# python
import os
import ast
import copy
import math
import pprint
from .six import PY2

# core
import maya.cmds as mc
import maya.mel as mel
import maya.api.OpenMaya as om
import pymel.core as pm

import fbx
import FbxCommon

# rbfSetup
if PY2:
    import rbf_io
    import rbf_node
else:
    from . import rbf_io
    from . import rbf_node

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from typing import (
            Text,
            Optional,
            Tuple,
            Dict,
            List,
        )
        Mode = Text


# ==============================================================================
# Constants
# ==============================================================================

CTL_SUFFIX = rbf_node.CTL_SUFFIX
DRIVEN_SUFFIX = rbf_node.DRIVEN_SUFFIX
TRANSFORM_SUFFIX = rbf_node.TRANSFORM_SUFFIX

WNODE_DRIVERPOSE_ATTRS = {"poseMatrix": "matrix",
                          "poseParentMatrix": "matrix",
                          "poseMode": "enum",
                          "controlPoseAttributes": "stringArray",
                          "controlPoseValues": "doubleArray",
                          "controlPoseRotateOrder": "enum"}

WNODE_TRANSFORM_ATTRS = ("tx",
                         "ty",
                         "tz",
                         "rx",
                         "ry",
                         "rz",
                         "sx",
                         "sy",
                         "sz",
                         "v")

WNODE_SHAPE_ATTRS = ("visibility",
                     "type",
                     "direction",
                     "invert",
                     "useRotate",
                     "angle",
                     "centerAngle",
                     "twist",
                     "twistAngle",
                     "useTranslate",
                     "grow",
                     "translateMin",
                     "translateMax",
                     "interpolation",
                     "iconSize",
                     "drawCone",
                     "drawCenterCone",
                     "drawWeight",
                     "outWeight",
                     "twistAxis",
                     "opposite",
                     "rbfMode",
                     "useInterpolation",
                     "allowNegativeWeights",
                     "scale",
                     "distanceType",
                     "drawOrigin",
                     "drawDriver",
                     "drawPoses",
                     "drawIndices",
                     "drawTwist",
                     "poseLength",
                     "indexDistance",
                     "driverIndex")

POSEDRIVER_ATTRS = ("translateX",
                    "translateY",
                    "translateZ",
                    "rotateX",
                    "rotateY",
                    "rotateZ")

REST_TRANSFORM_ATTR = "restTransforms"

ENVELOPE_ATTR = "scale"

WD_SUFFIX = "_PD"
RBF_TYPE = "poseDriver"
# RBFNode = "weightDriver"
DECOMPOSE_ROTATE_TYPE = "decomposeRotate"

# ==============================================================================
# General utils
# ==============================================================================
class PoseDriverNames():

    def __init__(self, src, dst):
        # type: (Text, Text) -> None

        self.src = src
        self.dst = dst
        self.driver = "{}_to_{}_posedriver_{}".format(src, dst, WD_SUFFIX)
        self.decompose_rotate = "{}_to_{}_posedriver_DR".format(src, dst)
        self.compose_matrix = "{}_to_{}_posedriver_CM".format(src, dst)


# Check for plugin
def loadWeightPlugin(dependentFunc):
    """ensure that plugin is always loaded prior to importing from json

    Args:
        dependentFunc (func): any function that needs to have plugin loaded

    Returns:
        func: pass through of function
    """
    try:
        pm.loadPlugin("weightDriver", qt=True)
    except RuntimeError:
        pm.displayWarning("RBF Manager couldn't found any valid RBF solver.")

    try:
        pm.loadPlugin("rotationDriver", qt=True)
    except RuntimeError:
        pm.displayWarning("Rotation Driver could not be loaded")

    return dependentFunc


def createRBF(name, transformName=None):
    # type: (Text, Optional[Text]) -> Tuple[pm.datatypes.Transform, pm.datatypes.WeightDriver]
    """Creates a rbf node of type weightDriver

    Args:
        name (str): name of node
        transformName (str, optional): specify name of transform

    Returns:
        list: pymel: trasnform, weightDriverShape
    """
    if transformName is None:
        transformName = "{}{}".format(name, TRANSFORM_SUFFIX)

    wd_ShapeNode = pm.createNode("weightDriver", n=name)
    wd_transform = pm.listRelatives(wd_ShapeNode, p=True)[0]
    wd_transform = pm.rename(wd_transform, transformName)
    pm.setAttr("{}.type".format(wd_ShapeNode), 1)

    return wd_transform, wd_ShapeNode


def __decomposeRotate(node, src, index):
    # type: (Text, Text, int) -> Text

    naming = PoseDriverNames(node, src)
    if not mc.objExists(naming.decompose_rotate):
        decomp = mc.createNode("decomposeRotate", n=naming.decompose_rotate)

    else:
        decomp = naming.decompose_rotate
        if __isConnectedWith(src, node, "decomposeRotate"):
            return decomp

    # in
    mc.connectAttr("{}.rotate".format(src), "{}.rotate".format(decomp), f=True)

    # out
    mc.connectAttr("{}.outRoll".format(decomp),  "{}.input[{}]".format(node, index * 3 + 0), f=True)
    mc.connectAttr("{}.outBendH".format(decomp), "{}.input[{}]".format(node, index * 3 + 1), f=True)
    mc.connectAttr("{}.outBendV".format(decomp), "{}.input[{}]".format(node, index * 3 + 2), f=True)

    return decomp


def __isConnectedWith(src, dst, interconnectorNodeType):
    # type: (Text, Text, Text) -> bool

    conn = mc.listConnections(src, s=False, d=True, scn=True, type=interconnectorNodeType, plugs=True)  # type: ignore
    conn = {x.split(".")[0] for x in conn}
    conn = mc.listConnections(conn, s=False, d=True, scn=True, plugs=True)  # type: ignore
    conn = {x.split(".")[0] for x in conn}

    return dst in conn


def __composeMatrix(node, dst, index):
    # type: (Text, Text, int) -> Text

    naming = PoseDriverNames(node, dst)
    if not mc.objExists(naming.compose_matrix):
        comp = mc.createNode("composeMatrix", n=naming.compose_matrix)

    else:
        comp = naming.compose_matrix
        if __isConnectedWith(node, dst, "composeMatrix"):
            return comp

    mc.connectAttr("{}.output[{}]".format(node, index * 7 + 0), "{}.inputTranslateX".format(comp), f=True)
    mc.connectAttr("{}.output[{}]".format(node, index * 7 + 1), "{}.inputTranslateY".format(comp), f=True)
    mc.connectAttr("{}.output[{}]".format(node, index * 7 + 2), "{}.inputTranslateZ".format(comp), f=True)
    # mc.connectAttr("{}.output[{}]".format(node, index * 6 + 3), "{}.inputRotateX".format(comp), f=True)
    # mc.connectAttr("{}.output[{}]".format(node, index * 6 + 4), "{}.inputRotateY".format(comp), f=True)
    # mc.connectAttr("{}.output[{}]".format(node, index * 6 + 5), "{}.inputRotateZ".format(comp), f=True)
    mc.setAttr("{}.useEulerRotation".format(comp), 0)
    mc.connectAttr("{}.output[{}]".format(node, index * 7 + 3), "{}.inputQuatX".format(comp))  # noqa
    mc.connectAttr("{}.output[{}]".format(node, index * 7 + 4), "{}.inputQuatY".format(comp))  # noqa
    mc.connectAttr("{}.output[{}]".format(node, index * 7 + 5), "{}.inputQuatZ".format(comp))  # noqa
    mc.connectAttr("{}.output[{}]".format(node, index * 7 + 6), "{}.inputQuatW".format(comp))  # noqa

    # set initial value of quat.w to 1.0 means euler(0, 0, 0) avoiding NaN
    mc.setAttr("{}.output[{}]".format(node, index * 7 + 6), 1.0)  # noqa

    mc.connectAttr(
        "{}.outputMatrix".format(comp),
        "{}.offsetParentMatrix".format(dst),
        f=True
    )

    return comp


def createPoseDriver(name, srcBones=None, driverBones=None, mode="rotation"):
    # type: (Text, Optional[List[Text]], Optional[List[Text]], Mode) -> Tuple[Text, Text]
    """Creates a PoseDriver nodes network of type weightDriver

    Args:

    Returns:
        list: pymel: trasnform, weightDriverShape
    """

    sel = mc.ls(sl=True) or []

    try:
        if srcBones is None:
            srcBones = [sel[0]]

        if driverBones is None:
            driverBones = [sel[1]]

    except IndexError:
        raise Exception("invalid argument or no selection to add_operator")

    naming = PoseDriverNames(name, srcBones[0])
    trn, ope = createRBF(naming.driver)

    for i, bone in enumerate(srcBones):
        if mode == "rotation":
            _ = __decomposeRotate(ope, bone, i)

    storeRestTransforms(ope.name(), driverBones)
    for i, bone in enumerate(driverBones):
        if mode == "rotation":
            _ = __composeMatrix(ope, bone, i)

    return trn, ope


def forceEvaluation(node):
    """force evaluation of the weightDriver node
    thank you Ingo

    Args:
        node (str): weightDriver to be recached
    """
    pm.setAttr("{}.evaluate".format(node), 1)


def getNodeConnections(node):
    """get all connections on weightDriver node

    Args:
        node (str): weightDriver node

    Returns:
        list: of connections and attrs to recreate,
        small list of supported nodes to be recreated
    """
    connections = []
    attributesToRecreate = []
    nodePlugConnections = pm.listConnections(node,
                                             plugs=True,
                                             scn=True,
                                             connections=True,
                                             sourceFirst=True)

    for connectPair in nodePlugConnections:
        srcPlug = connectPair[0].name()
        srcAttrName = connectPair[0].attrName(longName=True)
        destPlug = connectPair[1].name()
        destAttrName = connectPair[1].attrName(longName=True)
        connections.append([srcPlug, destPlug])
        # expand this list as we become more aware of the node
        if srcAttrName in ["solverGroupMessage"]:
            attributesToRecreate.append([srcPlug, "message"])
        if destAttrName in ["solverGroupMessage"]:
            attributesToRecreate.append([destPlug, "message"])
    return connections, attributesToRecreate


def getRBFTransformInfo(node):
    """get a dict of all the information to be serialized to/from json

    Args:
        node (str): name of weightDriverShape node

    Returns:
        dict: information to be recreated on import
    """
    tmp_dict = {}
    parentName = None
    nodeTransform = pm.listRelatives(node, p=True)[0]
    tmp_dict["name"] = nodeTransform.name()
    transformPar = pm.listRelatives(nodeTransform, p=True) or [None]
    if transformPar[0] is not None:
        parentName = transformPar[0].name()
    tmp_dict["parent"] = parentName
    for attr in WNODE_TRANSFORM_ATTRS:
        try:
            tmp_dict[attr] = nodeTransform.getAttr(attr)
        except AttributeError:
            pass
    return tmp_dict


def getIndexValue(nodePlug, indices):
    """return the values of a compound attr at the specified index

    Args:
        nodePlug (node.attr): name to compound attr
        indices (int): of the attr to get

    Returns:
        list: of indecies
    """
    allValues = []
    if indices:
        indices = list(range(indices[-1] + 1))
    for index in indices:
        attrPlugIdex = "{}[{}]".format(nodePlug, index)
        val = mc.getAttr(attrPlugIdex)
        allValues.append(val)
    return allValues


def lengthenCompoundAttrs(node):
    """In core, if a compound attr has a value of 0,0,0 it will skip creating
    the attribute. So to ensure that all indecies exist in the length of a
    compound we get fake get each index, forcing a create of that attr.

    # TODO Perhaps this can turned into a more useful function since we are
    already querying info that will be needed later on.

    Args:
        node (str): weightDriver to perform insanity check

    Returns:
        n/a: n/a
    """
    poseLen = mc.getAttr("{}.poses".format(node), mi=True)
    if poseLen is None:
        return
    attrSize = mc.getAttr("{}.input".format(node), s=True)
    valSize = mc.getAttr("{}.output".format(node), s=True)
    for poseIndex in poseLen:
        for index in range(attrSize):
            nodeInput = "{}.poses[{}].poseInput[{}]".format(node,
                                                            poseIndex,
                                                            index)
            mc.getAttr(nodeInput)

    for poseIndex in poseLen:
        for index in range(valSize):
            nodeValue = "{}.poses[{}].poseValue[{}]".format(node,
                                                            poseIndex,
                                                            index)
            mc.getAttr(nodeValue)


def getPoseInfo(node):
    # type: (Text) -> Dict[Text, Any]
    """Get dict of the pose info from the provided weightDriver node

    Args:
        node (str): name of weightDriver

    Returns:
        dict: of poseInput:list of values, poseValue:values
    """
    lengthenCompoundAttrs(node)
    tmp_dict = {"poseInput": [],
                "poseValue": []}
    numberOfPoses = pm.getAttr("{}.poses".format(node), mi=True) or []
    for index in numberOfPoses:
        nameAttrInput = "{0}.poses[{1}].poseInput".format(node, index)
        nameAttrValue = "{0}.poses[{1}].poseValue".format(node, index)
        poseInputIndex = pm.getAttr(nameAttrInput, mi=True) or []
        poseValueIndex = pm.getAttr(nameAttrValue, mi=True) or []
        poseInput = getIndexValue(nameAttrInput, poseInputIndex)
        poseValue = getIndexValue(nameAttrValue, poseValueIndex)
        tmp_dict["poseInput"].append(poseInput)
        tmp_dict["poseValue"].append(poseValue)

    return tmp_dict


def getDriverListInfo(node):
    """used for when live connections are supported on the weightDriver
    # TODO - Complete support

    Args:
        node (str): name of weightDriverNode

    Returns:
        dict: driver:poseValue
    """
    driver_dict = {}
    numberOfDrivers = pm.getAttr("{}.driverList".format(node), mi=True) or []
    for dIndex in numberOfDrivers:
        nameAttrDriver = "{0}.driverList[{1}].pose".format(node, dIndex)
        numberOfPoses = pm.getAttr(nameAttrDriver, mi=True) or []
        poseInfo = {}
        for pIndex in numberOfPoses:
            attrDriverPose = "{}[{}]".format(nameAttrDriver, pIndex)
            poseIndex = "pose[{}]".format(pIndex)
            tmp_dict = {}
            for key in WNODE_DRIVERPOSE_ATTRS.keys():
                attrValue = pm.getAttr("{}.{}".format(attrDriverPose, key))
                if type(attrValue) == pm.dt.Matrix:
                    attrValue = attrValue.get()
                tmp_dict[key] = attrValue
            poseInfo[poseIndex] = tmp_dict
        driver_dict["driverList[{}]".format(dIndex)] = poseInfo
    return driver_dict


def setDriverNode(node, driverNodes, mode):
    """set the node that will be driving the evaluation on our poses

    Args:
        node (str): name of weightDriver node
        driverNode (str): name of driver node
        driverAttrs (list): of attributes used to perform evaluations
    """
    print(f"setDriverNode({node=}, {driverNodes=})")

    for i, bone in enumerate(driverNodes):
        print(f"setDriverNode({bone=})")
        if mode == "rotation":
            _ = __decomposeRotate(node, bone, i)


def getDriverNode(node):
    """get nodes that are driving weightDriver node

    Args:
        node (str): weightDriver node

    Returns:
        list: of driver nodes
    """
    drivers = list(set(pm.listConnections("{}.input".format(node), scn=True)))
    if node in drivers:
        drivers.remove(node)
    drivers = {str(dNode.name()) for dNode in drivers if mc.ls(dNode.name(), showType=True)[1] == DECOMPOSE_ROTATE_TYPE}

    return list(drivers)


def setDrivenNode(node, drivenNodes):
    # type: (Text, List[Text]) -> List[Tuple[om.MVector, om.MVector]]
    """set the node to be driven by the weightDriver

    Args:
        node (str): weightDriver node
        drivenNodes (list): of nodes to be driven
    """

    vals = []
    for i, driven in enumerate(drivenNodes):

        val = getLocalXformWithOffsetparentmatrix(driven)
        vals.append(val)
        _ = __composeMatrix(node, driven, i)

    return vals


def storeRestTransforms(node, drivenBones=None):
    # type: (Text, Optional[List[Text]]) -> None
    """Stores rest transforms to the node."""

    if drivenBones is None:
        drivenBones = getDrivenNode(node)

    attrName = REST_TRANSFORM_ATTR
    attrType = "double"

    if not mc.objExists("{}.{}".format(node, attrName)):
        mc.addAttr(node, ln=attrName, at=attrType, multi=True)

    for i, driven in enumerate(drivenBones):
        t, r = getLocalXformWithOffsetparentmatrix(driven)
        print(f"storeRestTransforms({driven=}, {t=}, {r=})")
        mc.setAttr("{}.{}[{}]".format(node, attrName, i * 7 + 0), t[0])
        mc.setAttr("{}.{}[{}]".format(node, attrName, i * 7 + 1), t[1])
        mc.setAttr("{}.{}[{}]".format(node, attrName, i * 7 + 2), t[2])
        mc.setAttr("{}.{}[{}]".format(node, attrName, i * 7 + 3), r[0])
        mc.setAttr("{}.{}[{}]".format(node, attrName, i * 7 + 4), r[1])
        mc.setAttr("{}.{}[{}]".format(node, attrName, i * 7 + 5), r[2])
        mc.setAttr("{}.{}[{}]".format(node, attrName, i * 7 + 6), r[3])


def getRestTransforms(node):
    # type: (Text) -> List[Tuple[om.MVector, om.MEulerRotation]]
    values = mc.getAttr("{}.{}".format(node, REST_TRANSFORM_ATTR))[0]

    tx = None
    ty = None
    tz = None
    rx = None
    ry = None
    rz = None
    rw = None

    res = []
    for i, val in enumerate(values):

        if i % 7 == 0:
            tx = val
        elif i % 7 == 1:
            ty = val
        elif i % 7 == 2:
            tz = val
        elif i % 7 == 3:
            rx = val
        elif i % 7 == 4:
            ry = val
        elif i % 7 == 5:
            rz = val
        elif i % 7 == 6:
            rw = val

            t = (tx, ty, tz)
            r = (rx, ry, rz, rw)
            # t = om.MVector(tx, ty, tz)
            # r = om.MQuaternion(rx, ry, rz, rw)

            res.append((t, r))

    return res


def getLocalXformWithOffsetparentmatrix(target):
    # type: (Text) -> Tuple[om.MVector, om.MQuaternion]
    """"Manually calculate the local xfrom from the world matrix.

    Using the OffsetParentMatrix may prevent you from properly
    retrieving the local transform"""

    wm = mc.xform(target, q=True, ws=True, m=True)
    im = mc.getAttr("{}.parentInverseMatrix".format(target))

    wmm = om.MMatrix(wm)
    pmm = om.MMatrix(im)

    try:
        opm = om.MMatrix(mc.getAttr("{}.offsetParentMatrix".format(target)))
        res = om.MTransformationMatrix(wmm * pmm * opm)
    except ValueError:
        res = om.MTransformationMatrix(wmm * pmm)

    t = res.translation(om.MSpace.kWorld)
    r = res.rotation(asQuaternion=True)

    try:
        jo = mc.getAttr("{}.jointOrient".format(target))[0]
        jo = om.MEulerRotation(*[math.radians(x) for x in jo])
        r *= jo.inverse().asQuaternion()
    except ValueError:
        pass

    return (t, r)


def getDrivenNode(node):
    # type: (Text) -> List[Text]
    """get driven nodes connected to weightDriver

    Args:
        node (str): weightDriver node

    Returns:
        list: of driven nodes
    """
    driven = pm.listConnections("{}.output".format(node), scn=True)
    driven = list(set(pm.listConnections(driven, scn=True, s=False, d=True)))

    if node in driven:
        driven.remove(node)

    driven = [str(dNode.name()) for dNode in driven]

    return driven


def getSrcNode(node):
    # type: (pm.Transform) -> List[Text]
    """get driven nodes connected to weightDriver

    Args:
        node (str): weightDriver node

    Returns:
        list: str of src transform node
    """

    res = set(pm.listConnections(node, scn=True, s=True, d=False, type="decomposeRotate"))
    res = set(pm.listConnections(res, scn=True, s=True, d=False))

    if node in res:
        res.remove(node)

    if len(res) == 0:
        raise Exception("no composematrix node connected on node: {}".format(node))

    return [x.name() for x in res]


def getComposeMatrixNode(node):
    """get driven nodes connected to weightDriver

    Args:
        node (str): weightDriver node

    Returns:
        list: of driven nodes
    """
    driven = getDrivenNode(node)  # composematrix
    res = list({x for x in driven if mc.ls(x, showType=True)[1] == "composeMatrix"})

    if len(res) > 1:
        raise Exception("multiple composematrix nodes connected on node: {}".format(node))
    if len(res) == 0:
        raise Exception("no composematrix node connected on node: {}".format(node))

    return res[0]


def getDstNode(node):
    """get driven nodes connected to weightDriver

    Args:
        node (str): weightDriver node

    Returns:
        list: of driven nodes
    """
    comp = getComposeMatrixNode(node)
    res = list(set(pm.listConnections("{}.outputMatrix".format(comp), scn=True)))

    if len(res) > 1:
        raise Exception("multiple composematrix nodes connected on node: {}".format(node))
    if len(res) == 0:
        raise Exception("no composematrix node connected on node: {}".format(node))

    return res[0]


def getAttrInOrder(node, attrWithIndex):
    """get the connected attributes of the provided compound attr in order
    of index - Sanity check

    Args:
        node (str): weightDriver node
        attrWithIndex (str): name of compound attr with indicies to query

    Returns:
        list: of connected attrs in order
    """
    attrsToReturn = []
    attrs = mc.getAttr("{}.{}".format(node, attrWithIndex), mi=True) or []
    for index in attrs:
        nodePlug = "{}.{}[{}]".format(node, attrWithIndex, index)
        connected = pm.listConnections(nodePlug, scn=True, p=True)
        if not connected:
            continue
            connected = [None]
        attrsToReturn.append(connected[0])
    return attrsToReturn


def getDriverNodeAttributes(node):
    """get the connected attributes of the provided compound attr in order
    of index - Sanity check

    Args:
        node (str): weightDriver node

    Returns:
        list: of connected attrs in order
    """
    attributesToReturn = []
    driveAttrs = getAttrInOrder(node, "input")
    attributesToReturn = [attr.attrName(longName=True) for attr in driveAttrs
                          if attr.nodeName() != node]
    return attributesToReturn


def getDrivenNodeAttributes(node):
    """get the connected attributes of the provided compound attr in order
    of index - Sanity check

    Args:
        node (str): weightDriver node

    Returns:
        list: of connected attrs in order
    """
    attributesToReturn = []
    drivenAttrs = getAttrInOrder(node, "output")
    for attr in drivenAttrs:
        if attr.nodeName() != node:
            attrName = attr.getAlias() or attr.attrName(longName=True)
            attributesToReturn.append(attrName)
    return attributesToReturn


def copyPoses(nodeA, nodeB, emptyPoseValues=True):
    """Copy poses from nodeA to nodeB with the option to be blank or node
    for syncing nodes OF EQUAL LENGTH IN POSE INFO

    Args:
        nodeA (str): name of weightedNode
        nodeB (str): name of weightedNode
        emptyPoseValues (bool, optional): should the copy just be the same
        number of poses but blank output value

    Returns:
        n/a: n/a
    """
    posesIndices = pm.getAttr("{}.poses".format(nodeA), mi=True) or [None]
    if len(posesIndices) == 1 and posesIndices[0] is None:
        return
    nodeA_poseInfo = getPoseInfo(nodeA)
    drivenAttrs = getDrivenNodeAttributes(nodeB)
    nodeBdrivenIndex = list(range(len(drivenAttrs)))
    for attr, value in nodeA_poseInfo.items():
        if value == ():
            continue
        numberOfPoses = len(value)
        for poseIndex in range(numberOfPoses):
            poseValues = value[poseIndex]
            for index, pIndexValue in enumerate(poseValues):
                pathToAttr = "{}.poses[{}].{}[{}]".format(nodeB,
                                                          poseIndex,
                                                          attr,
                                                          index)
                if attr == "poseInput":
                    valueToSet = pIndexValue
                elif attr == "poseValue" and emptyPoseValues:
                    if drivenAttrs[index] in rbf_node.SCALE_ATTRS:
                        valueToSet = 1.0
                    else:
                        valueToSet = 0.0
                if index > nodeBdrivenIndex:
                    continue
                pm.setAttr(pathToAttr, valueToSet)


def syncPoseIndices(srcNode, destNode):
    """Syncs the pose indices between the srcNode and destNode.
    The input values will be copied from the srcNode, the poseValues will
    be defaulted to 0 or 1(if scaleAttr)

    Args:
        srcNode (str): weightedDriver
        destNode (str): weightedDriver
    """
    src_poseInfo = getPoseInfo(srcNode)
    destDrivenAttrs = getDrivenNodeAttributes(destNode)
    for poseIndex, piValues in enumerate(src_poseInfo["poseInput"]):
        for index, piValue in enumerate(piValues):
            pathToAttr = "{}.poses[{}].poseInput[{}]".format(destNode,
                                                             poseIndex,
                                                             index)
            pm.setAttr(pathToAttr, piValue)

    for poseIndex, piValues in enumerate(src_poseInfo["poseValue"]):
        for index, piValAttr in enumerate(destDrivenAttrs):
            pathToAttr = "{}.poses[{}].poseValue[{}]".format(destNode,
                                                             poseIndex,
                                                             index)
            if piValAttr in rbf_node.SCALE_ATTRS:
                valueToSet = 1.0
            else:
                valueToSet = 0.0
            pm.setAttr(pathToAttr, valueToSet)


def getNodeInfo(node):
    """get a dictionary of all the serialized information from the desired
    weightDriver node for export/import/duplication

    Args:
        node (str): name of weightDriver node

    Returns:
        dict: collected node info
    """

    if isinstance(node, str):
        node = pm.PyNode(node)
    # node = pm.PyNode(node)
    weightNodeInfo_dict = {}
    for attr in WNODE_SHAPE_ATTRS:
        weightNodeInfo_dict[attr] = node.getAttr(attr)
    weightNodeInfo_dict["transformNode"] = getRBFTransformInfo(node)
    connections, attributesToRecreate = getNodeConnections(node)
    weightNodeInfo_dict["connections"] = connections
    weightNodeInfo_dict["attributesToRecreate"] = attributesToRecreate
    weightNodeInfo_dict["poses"] = getPoseInfo(node)

    # is an attribute on the weightedDriver node
    weightNodeInfo_dict["driverList"] = getDriverListInfo(node)

    # actual source node that is driving the poses on node
    weightNodeInfo_dict["driverNode"] = getSrcNode(node)

    # attr on driver node pushing the poses
    weightNodeInfo_dict["driverAttrs"] = getDriverNodeAttributes(node)

    # node being driven by the setup
    weightNodeInfo_dict["drivenNode"] = getDrivenNode(node)

    # node.attrs being driven by the setup
    weightNodeInfo_dict["drivenAttrs"] = getDrivenNodeAttributes(node)

    driverContol = rbf_node.getDriverControlAttr(node.name())
    weightNodeInfo_dict["driverControl"] = driverContol

    weightNodeInfo_dict["setupName"] = rbf_node.getSetupName(node.name())

    drivenControlName = rbf_node.getConnectedRBFToggleNode(node.name(),
                                                           ENVELOPE_ATTR)
    weightNodeInfo_dict["drivenControlName"] = drivenControlName

    weightNodeInfo_dict["rbfType"] = RBF_TYPE

    driverPosesInfo = rbf_node.getDriverControlPoseAttr(node.name())
    weightNodeInfo_dict[rbf_node.DRIVER_POSES_INFO_ATTR] = driverPosesInfo

    # rest tranforms of driven nods
    weightNodeInfo_dict[REST_TRANSFORM_ATTR] = getRestTransforms(node.name())
    return weightNodeInfo_dict


def setTransformNode(transformNode, transformInfo):
    """set the transform node of a weightedDriver with the information from
    dict

    Args:
        transformNode (str): name of transform nodes
        transformInfo (dict): information to set on transform node
    """
    parent = transformInfo.pop("parent", None)
    if parent is not None:
        pm.parent(transformNode, parent)
    for attr, value in transformInfo.items():
        # transformNode.setAttr(attr, value)
        pm.setAttr("{}.{}".format(transformNode, attr), value)


def deletePose(node, indexToPop):
    # type: (Text, int) -> None
    """gather information on node, remove desired index and reapply

    Args:
        node (str): weightDriver
        indexToPop (int): pose index to remove
    """
    node = pm.PyNode(node)
    posesInfo = getPoseInfo(node)
    poseInput = posesInfo["poseInput"]
    poseValue = posesInfo["poseValue"]
    currentLength = list(range(len(poseInput)))
    poseInput.pop(indexToPop)
    poseValue.pop(indexToPop)
    setPosesFromInfo(node, posesInfo)
    attrPlug = "{}.poses[{}]".format(node, currentLength[-1])
    pm.removeMultiInstance(attrPlug, b=True)


def addPose(node, poseInput, poseValue, posesIndex=None):
    """add pose to the weightDriver node provided. Also used for editing an
    existing pose, since you can specify the index. If non provided assume new

    Args:
        node (str): weightedDriver
        poseInput (list): list of the poseInput values
        poseValue (list): of poseValue values
        posesIndex (int, optional): at desired index, if none assume latest/new
    """
    print(f"addPose({node=}, {poseInput=}, {poseValue}")
    if posesIndex is None:
        posesIndex = len(pm.getAttr("{}.poses".format(node), mi=True) or [])

    for index, value in enumerate(poseInput):
        print(f"addPose({index}, input: {value=}")
        attrPlug = "{}.poses[{}].poseInput[{}]".format(node, posesIndex, index)
        pm.setAttr(attrPlug, value)

    for index, value in enumerate(poseValue):
        print(f"addPose({index}, pose: {value=}")
        attrPlug = "{}.poses[{}].poseValue[{}]".format(node, posesIndex, index)
        pm.setAttr(attrPlug, value)


def setPosesFromInfo(node, posesInfo):
    """set a large number of poses from the dictionary provided

    Args:
        node (str): weightDriver
        posesInfo (dict): of poseInput/PoseValue:values
    """
    for attr, value in posesInfo.items():
        if value == ():
            continue
        numberOfPoses = len(value)
        for poseIndex in range(numberOfPoses):
            poseValues = value[poseIndex]
            for index, pIndexValue in enumerate(poseValues):
                pathToAttr = "poses[{}].{}[{}]".format(poseIndex,
                                                       attr,
                                                       index)
                node.setAttr(pathToAttr, pIndexValue)


def setPose(node, posesIndex):
    """set a large number of poses from the dictionary provided

    Args:
        node (str): weightDriver
        posesInfo (dict): of poseInput/PoseValue:values
    """
    poseInfo = rbf_node.getDriverControlPoseAttr(node)

    for attr, value in posesInfo.items():
        if value == ():
            continue
        numberOfPoses = len(value)
        for poseIndex in range(numberOfPoses):
            poseValues = value[poseIndex]
            for index, pIndexValue in enumerate(poseValues):
                pathToAttr = "poses[{}].{}[{}]".format(poseIndex,
                                                       attr,
                                                       index)
                node.setAttr(pathToAttr, pIndexValue)


def setDriverListFromInfo(node, driverListInfo):
    """set driverlist node with information from dict proivided

    Args:
        node (pynode): name of driver node
        driverListInfo (dict): attr/value
    """
    for attr, posesInfo in driverListInfo.items():
        # attrDriver = "{}.pose".format(attr)
        numberOfPoses = len(posesInfo.keys())
        for pIndex in range(numberOfPoses):
            poseIndex = "pose[{}]".format(pIndex)
            poseAttrIndex = "{}.{}".format(attr, poseIndex)
            for driverAttr, attrType in WNODE_DRIVERPOSE_ATTRS.items():
                fullPathToAttr = "{}.{}".format(poseAttrIndex, driverAttr)
                attrValue = posesInfo[poseIndex][driverAttr]
                if attrType == "enum":
                    node.setAttr(fullPathToAttr, attrValue)
                elif attrType == "matrix":
                    attrValue = pm.dt.Matrix(attrValue)
                    node.setAttr(fullPathToAttr, attrValue, type=attrType)
                else:
                    node.setAttr(fullPathToAttr, attrValue, type=attrType)


def setWeightNodeAttributes(node, weightNodeAttrInfo):
    """set the attribute information on the weightDriver node provided from
    the info dict

    Args:
        node (pynode): name of weightDrivers
        weightNodeAttrInfo (dict): of attr:value
    """
    failedAttrSets = []
    for attr, value in weightNodeAttrInfo.items():
        try:
            pm.setAttr("{}.{}".format(node, attr), value)
        except Exception as e:
            failedAttrSets.append([attr, value, e])
    if failedAttrSets:
        pprint.pprint(failedAttrSets)


def createVectorDriver(driverInfo):
    # future vector driver support starts here
    pass


def recreateAttributes(node, attributesToRecreate):
    """add any attributes to the provided node from list

    Args:
        node (str): name of node
        attributesToRecreate (list): of attrs to add
    """
    for attrInfo in attributesToRecreate:
        attrPlug = attrInfo[0]
        attrType = attrInfo[1]
        attrName = attrPlug.split(".")[1]
        if pm.objExists(attrPlug):
            continue
        pm.addAttr(node, ln=attrName, at=attrType)


def recreateConnections(connectionsInfo):
    """recreate connections from dict

    Args:
        connectionsInfo (dict): of nodes.attr plugs to try and recreate
    """
    failedConnections = []
    for attrPair in connectionsInfo:
        try:
            pm.connectAttr(attrPair[0], attrPair[1], f=True)
        except Exception as e:
            failedConnections.append([attrPair, e])
    if failedConnections:
        print("The Following Connections failed...")
        pprint.pprint(failedConnections)


def updateDriverControlPoseAttr(node, driverControls, poseIndex):
    # type: (Text, List[Text], int) -> None
    """get the ControlPoseDict add any additionally recorded values to and set

    Args:
        node (str): name of the RBFNode supported node
        driverControls (list): name of the control to queary attr info from
        poseIndex (int): to add the collected pose information to
    """

    poseInfos = rbf_node.getDriverControlPoseAttr(node)
    if not poseInfos:
        poseInfos = []

    print(f"updateDriverControlPoseAttr: {poseIndex=}: {driverControls=}, ")
    for i, driven in enumerate(driverControls):
        print(f"updateDriverControlPoseAttr: {i}: {driven=}, ")

        try:
            poseInfo = poseInfos[i]
        except IndexError:
            poseInfo = {}
            poseInfos.append(poseInfo)

        t, q = getLocalXformWithOffsetparentmatrix(driven)
        r = om.MQuaternion(q).asEulerRotation()

        for attr, val in zip(
            ("translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ"),
            (t[0], t[1], t[2], r[0], r[1], r[2])
        ):
            attrPoseIndices = poseInfo.get(attr, [])
            lengthOfList = len(attrPoseIndices) - 1

            if not attrPoseIndices or lengthOfList < poseIndex:
                attrPoseIndices.insert(poseIndex, val)
            elif lengthOfList >= poseIndex:
                attrPoseIndices[poseIndex] = val

            poseInfo[attr] = attrPoseIndices

    rbf_node.setDriverControlPoseAttr(node, poseInfos)


@loadWeightPlugin
def createRBFFromInfo(weightNodeInfo_dict):
    """create an rbf node from the dictionary provided information

    Args:
        weightNodeInfo_dict (dict): of weightDriver information

    Returns:
        list: of all created weightDriver nodes
    """
    createdNodes = []
    skipped_nodes = []
    weightNodeInfo_dict = copy.deepcopy(weightNodeInfo_dict)
    for weightNodeName, weightInfo in weightNodeInfo_dict.items():
        rbfType = weightInfo.pop("rbfType", RBF_TYPE)
        connectionsInfo = weightInfo.pop("connections", {})
        posesInfo = weightInfo.pop("poses", {})
        transformNodeInfo = weightInfo.pop("transformNode", {})
        driverListInfo = weightInfo.pop("driverList", {})
        attributesToRecreate = weightInfo.pop("attributesToRecreate", [])
        # hook for future support of vector
        driverInfo = weightInfo.pop("vectorDriver", {})
        driverNodeName = weightInfo.pop("driverNode", {})
        driverNodeAttributes = weightInfo.pop("driverAttrs", {})
        drivenNodeName = weightInfo.pop("drivenNode", {})
        drivenNodeAttributes = weightInfo.pop("drivenAttrs", {})
        transformName = transformNodeInfo.pop("name", None)
        setupName = weightInfo.pop("setupName", "")
        drivenControlName = weightInfo.pop("drivenControlName", "")
        driverControl = weightInfo.pop("driverControl", "")
        driverControlPoseInfo = weightInfo.pop(rbf_node.DRIVER_POSES_INFO_ATTR,
                                               {})

        # if not mc.objExists(drivenControlName):
        #     skipped_nodes.append(drivenControlName)
        #     continue
        node = RBFNode(weightNodeName)

        node.create(driverNodeName, drivenNodeName)
        node.setSetupName(setupName)
        node.setDriverControlAttr(drivenNodeName)
        node.setDriverNode(driverNodeName, driverNodeAttributes)
        pynode = pm.PyNode(node.name)

        setWeightNodeAttributes(pynode, weightInfo)
        recreateAttributes(pynode, attributesToRecreate)
        setPosesFromInfo(pynode, posesInfo)
        setDriverListFromInfo(pynode, driverListInfo)
        rbf_node.setDriverControlPoseAttr(node.name, driverControlPoseInfo)
        # recreateConnections(connectionsInfo)
        createdNodes.append(node)

    if skipped_nodes:
        mc.warning("RBF Nodes were skipped due to missing controls! \n {}".format(skipped_nodes))

    return createdNodes


def getNodesInfo(weightDriverNodes):
    """convenience function to get a dict of all the provided nodes

    Args:
        weightDriverNodes (list): names of all weightDriver nodes

    Returns:
        dict: collected serialized informtiaon
    """
    weightNodeInfo_dict = {}
    for wdNode in weightDriverNodes:
        wdNode = pm.PyNode(wdNode)
        weightNodeInfo_dict[wdNode.name()] = getNodeInfo(wdNode)
    return weightNodeInfo_dict


def exportNodes(filePath, weightDriverNodes):
    """export serialized node information to the specified filepath

    Args:
        filePath (str): path/to/file.ext
        weightDriverNodes (list): of weightDriver nodes
    """
    weightNodeInfo_dict = getNodesInfo(weightDriverNodes)
    rbf_io._exportData(weightNodeInfo_dict, filePath)
    print("Weight Driver Nodes successfully exported: {}".format(filePath))


def importNodes(filePath):
    """create nodes from serialized data from the provided json filepath

    Args:
        filePath (str): path/to/file
    """
    weightNodeInfo_dict = rbf_io._importData(filePath)
    createRBFFromInfo(weightNodeInfo_dict)


def exportRBFs(nodes, filePath):
    """exports the desired rbf nodes to the filepath provided

    Args:
        nodes (list): of rbfnodes
        filePath (str): filepath to json
    """
    rbfNode_Info = {}
    for n in nodes:
        rbfNode_Info[n] = getNodeInfo(n)
    rbf_io.__exportData(rbfNode_Info, filePath)
    print("RBF Data exported: {}".format(filePath))

    exportPoseAsFBX(nodes, filePath.replace("rbf", "fbx"))


def exportPoseAsFBX(nodes, filePath):
    # type: (List[pm.PyNode], str) -> None
    """exports the pose to fbx for UE posedriver."""
    mc.select(cl=True)

    for node in nodes:
        nodeInfo = getNodeInfo(node)

        driver = nodeInfo.get("driverNode", "")
        driven = nodeInfo.get("drivenNode", "")
        poses = nodeInfo.get("poses", {})
        input_ = poses.get("poseInput", [])
        value = poses.get("poseValue", [])
        rest = nodeInfo.get(REST_TRANSFORM_ATTR , [])

        keyInfo = {}
        for i, bone in enumerate(driver):
            keyInfo[bone] = {"rotation": [], "translation": []}
            for pi in input_:
                rs = i * 3 + 0
                re = i * 3 + 3
                rot = pi[rs:re]
                keyInfo[bone]["rotation"].append(rot)

        for i, bone in enumerate(driven):
            keyInfo[bone] = {"rotation": [], "translation": []}
            for pi in value:
                ts = i * 7 + 0
                te = i * 7 + 3
                rs = i * 7 + 3
                re = i * 7 + 8
                rot = pi[rs:re]
                tra = pi[ts:te]

                tra[0] += rest[i][0][0]
                tra[1] += rest[i][0][1]
                tra[2] += rest[i][0][2]
                rot[0] += rest[i][1][0]
                rot[1] += rest[i][1][1]
                rot[2] += rest[i][1][2]
                keyInfo[bone]["rotation"].append(rot)
                keyInfo[bone]["translation"].append(tra)

        mc.select(mc.ls(driver))
        mc.select(mc.ls(driven), add=True)

        exportFBX(nodeInfo.get("setupName"), filePath, keyInfo)


def exportFBX(name, fbxPath, keyInfo):
    safePath = fbxPath.replace(os.sep, "/")
    mel.eval("FBXResetExport")
    # mel.eval('FBXLoadExportPresetFile -f "{}";'.format(preset_path))

    # frame range ---------------------------------------------------
    mel.eval("FBXExportBakeComplexAnimation -v 1")
    # if bake_animation:
    #     mc.playbackOptions(min=frame_start)
    #     mc.playbackOptions(max=frame_end)
    mel.eval("FBXExportBakeComplexStart -v 0")
    mel.eval("FBXExportBakeComplexEnd -v 0")
    mel.eval("FBXExportSplitAnimationIntoTakes -clear;")
    cmd = u"""FBXExportSplitAnimationIntoTakes -v "{}" {} {};""".format(name, 0, len(keyInfo))
    mel.eval(cmd)
    mel.eval("FBXExportDeleteOriginalTakeOnSplitAnimation - v true;")

    # do export -----------------------------------------------------
    mel.eval(u'FBXExport -s -f "{}";'.format(safePath))
    mel.eval("FBXExportSplitAnimationIntoTakes -clear;")

    # --------------------------------------------------------------
    modify_fbx_file_to_add_key(fbxPath, keyInfo)



class RBFNode(rbf_node.RBFNode):
    """when subclassed everything that need be overrided is information
    specific to the module rbf node.

    Attributes:
        name (str): name of the node that either exists or to be created
        rbfType (str): nodeType to create node of supported type
        transformNode (str): name of transform node
    """

    def __init__(self, name, src=None, dst=None):

        self.rbfType = RBF_TYPE
        self.name = name
        self.transformNode = None

        if mc.objExists(name) and mc.nodeType(name) in rbf_node.SUPPORTED_RBF_NODES:
            self.rbfType = mc.nodeType(name)
            self.transformNode = self.getTransformParent()
            self.lengthenCompoundAttrs()
            self.restoreAttributeFromeNode(name)
        else:
            pass

    def restoreAttributeFromeNode(self, name):
        # type: (Text) -> None
        """restore attributes by retrieving connections"""

        self.rbfType = RBF_TYPE
        self.transformNode = self.getTransformParent()
        self.lengthenCompoundAttrs()
        self.src = getSrcNode(name)
        self.dst = getDrivenNode(name)

    def nodeType_suffix(self):
        return WD_SUFFIX

    def create(self, srcBones, driveBones):
        # type: (List[Text], List[Text]) -> None
        name = self.formatName(self.name, self.nodeType_suffix())
        transformNode, node = createPoseDriver(name, srcBones, driveBones)
        self.transformNode = transformNode.name()
        self.name = node.name()

        self.src = getSrcNode(node)
        self.dst = getDrivenNode(node)

    def getPoseInfo(self):
        return getPoseInfo(self.name)

    def getNodeInfo(self):
        return getNodeInfo(pm.PyNode(self.name))

    def lengthenCompoundAttrs(self):
        lengthenCompoundAttrs(self.name)

    def addPose(self, poseInput=None, poseValue=None, posesIndex=None):
        # type: (Optional[List[float]], Optional[List[float]], Optional[int]) -> None

        if poseInput is None:
            driverNode = self.getDriverNode()[0]
            driverAttrs = self.getDriverNodeAttributes()
            poseInput = rbf_node.getMultipleAttrs(driverNode, driverAttrs)

        if poseValue is None:
            poseValue = self.getPoseValues()

        if posesIndex is None:
            posesIndex = len(self.getPoseInfo()["poseInput"])

        self.updateDriverControlPoseAttr(posesIndex)
        addPose(self.name,
                poseInput,
                poseValue,
                posesIndex=posesIndex)

        self.resetToRest()

    def resetToRest(self):
        # type: () -> None
        restXforms = getRestTransforms(self.name)
        drivenNodes = self.getDrivenNode()
        for xform, driven in zip(restXforms, drivenNodes):
            mc.setAttr("{}.translate".format(driven), *xform[0])
            mc.setAttr("{}.rotate".format(driven), *xform[1])

    def deletePose(self, indexToPop):
        # type: (int) -> None
        deletePose(self.name, indexToPop)

    def getDriverNode(self):
        return getDriverNode(self.name)

    def getDriverNodeAttributes(self):
        return getDriverNodeAttributes(self.name)

    def getDrivenNode(self):
        return getDrivenNode(self.name)

    def getDrivenNodeAttributes(self):
        return getDrivenNodeAttributes(self.name)

    def getTransformParent(self):
        return getRBFTransformInfo(self)["name"]

    def getSrcNode(self):
        return getRBFTransformInfo(self)["src"]

    def setDriverNode(self, driverNode, driverAttrs):
        setDriverNode(self.name, driverNode, driverAttrs)
        self.src = driverNode

    def setDrivenNode(self, drivenNodes):
        # type: (List[Text]) -> Dict[Text, Text]

        currentXforms = setDrivenNode(self.name, drivenNodes)

        for drivenNode in drivenNodes:
            if drivenNode.endswith(DRIVEN_SUFFIX):

                rbf_node.createRBFToggleAttr(drivenNode)
                rbf_node.connectRBFToggleAttr(drivenNode,
                                              self.name,
                                              self.getRBFToggleAttr())
        return currentXforms

    def copyPoses(self, nodeB):
        poseInfo = self.getDriverControlPoseAttr()
        nodeB.setDriverControlPoseAttr(poseInfo)
        copyPoses(self.name, nodeB)

    def forceEvaluation(self):
        forceEvaluation(self.transformNode)

    def getRBFToggleAttr(self):
        return ENVELOPE_ATTR

    def syncPoseIndices(self, srcNode):
        poseInfo = srcNode.getDriverControlPoseAttr()
        self.setDriverControlPoseAttr(poseInfo)
        syncPoseIndices(srcNode, self.name)

    def updateDriverControlPoseAttr(self, posesIndex):
        """update the driverControlPoseAttr at the specified index

        Args:
            posesIndex (int): update the pose information at the index
        """
        driverControl = self.getDriverControlAttr()
        updateDriverControlPoseAttr(self.name, driverControl, posesIndex)

    def getDriverControlAttr(self):
        # type: () -> Text
        res = super(RBFNode, self).getDriverControlAttr()
        return ast.literal_eval(res)

    def getPose(self, posesIndex):
        # type: (int) -> List[float]
        return []

    def getPoseValues(self, resetDriven=True, absoluteWorld=True):
        # type: (bool, bool) -> List[float]
        """get all pose values from rbf node

        Args:
            resetDriven (bool, optional): reset driven animControl

        Returns:
            list: of poseValues
        """

        res = []
        drivenNodes = self.getDrivenNode()
        restXforms = getRestTransforms(self.name)
        for (restT, restR), node in zip(restXforms, drivenNodes):

            rt = om.MVector(restT)
            rr = om.MQuaternion(restR)

            t, r = getLocalXformWithOffsetparentmatrix(node)
            t -= rt
            r *= (rr.inverse())

            res.append(t[0])
            res.append(t[1])
            res.append(t[2])
            res.append(r[0])
            res.append(r[1])
            res.append(r[2])
            res.append(r[3])

        return res

    def recallSourcePose(self, posesIndex):

        for i, source in enumerate(self.src):

            pathToAttr = "{}.poses[{}].poseInput".format(self.name, posesIndex)
            pose = mc.getAttr(pathToAttr)[i]
            tmp = mc.createNode("composeRotate")
            mc.setAttr("{}.decomposedAngle.roll".format(tmp), pose[0])
            mc.setAttr("{}.decomposedAngle.bendH".format(tmp), pose[1])
            mc.setAttr("{}.decomposedAngle.bendV".format(tmp), pose[2])
            r = mc.getAttr("{}.outRotate".format(tmp))[0]
            mc.setAttr("{}.rotateX".format(source), r[0])
            mc.setAttr("{}.rotateY".format(source), r[1])
            mc.setAttr("{}.rotateZ".format(source), r[2])
            mc.delete(tmp)

    def recallDriverPose(self, posesIndex):
        # type: (int) -> None

        self.recallSourcePose(posesIndex)
        driverControl = self.getDriverControlAttr()
        poseInfos = rbf_node.getDriverControlPoseAttr(self.name)
        # info = poseInfos[posesIndex]

        for attrDict, driven in zip(poseInfos, driverControl):

            # restore from poseinfo
            for attrName, values in attrDict.items():
                try:
                    attrPath = "{}.{}".format(driven, attrName)
                    mc.setAttr(attrPath, values[posesIndex])
                    print(f"{attrPath=}, {values[posesIndex]=}")

                except RuntimeError:
                    print(f"setAttr failed: {attrPath=}, {values[posesIndex]=}")

            # cancel offset parent matrix of current poseDriver's effect
            opm = om.MMatrix(mc.getAttr("{}.offsetParentMatrix".format(driven)))

            lm = om.MTransformationMatrix()
            t = om.MVector(mc.getAttr("{}.translate".format(driven))[0])
            r = om.MEulerRotation(mc.getAttr("{}.rotate".format(driven))[0])
            lm.setTranslation(t, om.MSpace.kObject)
            lm.setRotation(r)
            # print(f"{driven=}, local: {t=}, {r=}")

            tm = om.MTransformationMatrix(lm.asMatrix() * opm.inverse())
            t = tm.translation(om.MSpace.kObject)
            r = tm.rotation()

            mc.setAttr("{}.translate".format(driven), *t)
            mc.setAttr("{}.rotate".format(driven), *r)
            # print(f"{driven=}, offset: {t=}, {r=}")


# ----------------------------------------------------------------------------
# FBX Utility
# ----------------------------------------------------------------------------
def modify_fbx_file_to_add_key(fbx_path, key_info):
    # type: (Text, Dict[Text, Dict[Text, List[List[float]]]]) -> None
    """modify fbx file to add key."""
    sdk_manager, fbx_scene = FbxCommon.InitializeSdkObjects()

    result = FbxCommon.LoadScene(sdk_manager, fbx_scene, fbx_path)
    if not result:
        print("An error occurred while loading the scene...")
        return

    _root = fbx_scene.GetRootNode()
    time = fbx.FbxTime(0)
    for node, attr_keys in key_info.items():

        for attr, keys in attr_keys.items():
            curve_x = __get_or_create_curve(fbx_scene, node, attr, "X")
            curve_y = __get_or_create_curve(fbx_scene, node, attr, "Y")
            curve_z = __get_or_create_curve(fbx_scene, node, attr, "Z")

            curve_x.KeyModifyBegin()
            curve_y.KeyModifyBegin()
            curve_z.KeyModifyBegin()

            for i, value in enumerate(keys):
                time.SetFrame(i)
                __add_key_to_curve(curve_x, time, value[0])
                __add_key_to_curve(curve_y, time, value[1])
                __add_key_to_curve(curve_z, time, value[2])

            curve_x.KeyModifyEnd()
            curve_y.KeyModifyEnd()
            curve_z.KeyModifyEnd()

    anim_stack = fbx_scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    span = fbx.FbxTimeSpan(fbx.FbxTime(0), time)
    anim_stack.SetLocalTimeSpan(span)
    global_settings = fbx_scene.GetGlobalSettings()
    global_settings.SetTimelineDefaultTimeSpan(span)

    FbxCommon.SaveScene(sdk_manager, fbx_scene, fbx_path)

    sdk_manager.Destroy()


def __get_or_create_curve(scene, target_name, attr, component):
    # type: (fbx.FbxScene, Text, Text, Text) -> fbx.FbxAnimCurve
    """Get or create curve."""

    anim_stacks = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
    if anim_stacks == 0:
        raise Exception("no animation stack found")

    node = scene.FindNodeByName(target_name)
    if not node:
        raise Exception("node not found: %s" % target_name)

    # anim stack means animation clip
    anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    anim_layers = anim_stack.GetMemberCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId))
    if anim_layers == 0:
        print("no animation layer found")
        return
    if anim_layers > 1:
        print("multiple animation layer found")

    anim_layer = anim_stack.GetMember(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)

    if attr.lower() == "translation":
        curve = node.LclTranslation.GetCurve(anim_layer, component, True)
    elif attr.lower() == "rotation":
        curve = node.LclRotation.GetCurve(anim_layer, component, True)
    else:
        raise Exception("invalid attr %s", attr)

    if not curve:
        raise Exception("curve not found on %s, %s" % target_name, attr)

    return curve


def __add_key_to_curve(curve, time, value):
    # type: (fbx.FbxAnimCurve, fbx.FbxTime, float) -> None
    """Add key to curve."""

    key = curve.KeyAdd(time)
    curve.KeySet(key[0], time, value)
