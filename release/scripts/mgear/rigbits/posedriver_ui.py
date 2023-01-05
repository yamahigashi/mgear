#!/usr/bin/env python
"""
A tool to manage a number of rbf type nodes under a user defined setup(name)

Steps -
    set Driver
    set Control for driver(optional, recommended)
    select attributes to driver RBF nodes
    Select Node to be driven in scene(Animation control, transform)
    Name newly created setup
    select attributes to be driven by the setup
    add any additional driven nodes
    position driver(via the control)
    position the driven node(s)
    select add pose

Add notes -
Please ensure the driver node is NOT in the same position more than once. This
will cause the RBFNode to fail while calculating. This can be fixed by deleting
any two poses with the same input values.

Edit Notes -
Edit a pose by selecting "pose #" in the table. (which recalls recorded pose)
reposition any controls involved in the setup
select "Edit Pose"

Delete notes -
select desired "pose #"
select "Delete Pose"

Mirror notes -
setups/Controls will succefully mirror if they have had their inverseAttrs
configured previously.

2.0 -------
LOOK into coloring the pose and how close it is
import replace name support (will work through json manually)
support live connections
settings support for suffix, etc
rename existing setup
newScene callback

Attributes:
    CTL_SUFFIX (str): suffix for anim controls
    DRIVEN_SUFFIX (str): suffix for driven group nodes
    EXTRA_MODULE_DICT (str): name of the dict which holds additional modules
    MGEAR_EXTRA_ENVIRON (str): environment variable to query for paths
    TOOL_NAME (str): name of UI
    TOOL_TITLE (str): title as it appears in the ui
    __version__ (float): UI version

Deleted Attributes:
    RBF_MODULES (dict): of supported rbf modules

__author__ = "Takayoshi Matsumoto"
__email__ = "yamahigashi@gmail.com"
__credits__ = ["Rafael Villar", "Miquel Campos", "Ingo Clemens"]

"""
# python
import imp
import os
import sys
from functools import partial

# core
import maya.cmds as mc
import pymel.core as pm
import maya.OpenMaya as om
import maya.OpenMayaUI as mui
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin

# mgear
import mgear
from mgear.core import pyqt
import mgear.core.string as mString
from mgear.core import anim_utils

from mgear.vendor.Qt import (
    QtWidgets,  # type: ignore
    QtCore,  # type: ignore
    QtCompat,  # type: ignore
)


if sys.version_info > (3, 0):
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from typing import (
            Text,  # noqa
            List,  # noqa
            Tuple,  # noqa
        )
        from mgear.vendor.Qt import QtGui  # type: ignore  # noqa

# rbf
from . import rbf_io
from . import rbf_node
from . import rbf_manager_ui
from .posedriver_io import (
    RBFNode,  # noqa
    exportRBFs,  # noqa
)# noqa

# from . import weightNode_io
from .six import PY2

from logging import getLogger, WARN, DEBUG, INFO  # noqa

logger = getLogger(__name__)
logger.setLevel(INFO)

# debug
# reload(rbf_io)
# reload(rbf_node)

# =============================================================================
# Constants
# =============================================================================
__version__ = "0.1.0"

_mgear_version = mgear.getVersion()
TOOL_NAME = "PoseDriver Manager"
TOOL_TITLE = "{} v{} | mGear {}".format(TOOL_NAME, __version__, _mgear_version)

DRIVEN_SUFFIX = rbf_node.DRIVEN_SUFFIX
CTL_SUFFIX = rbf_node.CTL_SUFFIX

MGEAR_EXTRA_ENVIRON = "MGEAR_RBF_EXTRA"
EXTRA_MODULE_DICT = "extraFunc_dict"

MIRROR_SUFFIX = "_mr"

# =============================================================================
# general functions
# =============================================================================


def testFunctions(*args):
    """test function for connecting signals during debug

    Args:
        *args: Description
    """
    print("!!", args)


def getPlugAttrs(nodes, attrType="all"):
    """Get a list of attributes to display to the user

    Args:
        nodes (str): name of node to attr query
        keyable (bool, optional): should the list only be kayable attrs

    Returns:
        list: list of attrplugs
    """
    plugAttrs = []
    for node in nodes:
        if attrType == "all":
            attrs = mc.listAttr(node, se=True, u=False)
            aliasAttrs = mc.aliasAttr(node, q=True)
            if aliasAttrs is not None:
                try:
                    attrs.extend(aliasAttrs[0::2])
                except Exception:
                    pass
        elif attrType == "cb":
            attrs = mc.listAttr(node, se=True, u=False, cb=True)
        elif attrType == "keyable":
            attrs = mc.listAttr(node, se=True, u=False, keyable=True)
        if attrs is None:
            continue
        [plugAttrs.append("{}.{}".format(node, a)) for a in attrs]
    return plugAttrs


def existing_rbf_setup(node):
    """check if there is an existing rbf setup associated with the node

    Args:
        node (str): name of the node to query

    Returns:
        list: of the rbftype assiociated with the node
    """
    connected_nodes = (
        mc.listConnections(node, destination=True, shapes=True, scn=True) or []
    )
    connected_node_types = set(mc.nodeType(x) for x in connected_nodes)
    rbf_node_types = set(rbf_io.RBF_MODULES.keys())
    return list(connected_node_types.intersection(rbf_node_types))


def getEnvironModules():
    """if there are any environment variables set that load additional
    modules for the UI, query and return dict

    Returns:
        dict: displayName:funcObject
    """
    extraModulePath = os.environ.get(MGEAR_EXTRA_ENVIRON, None)
    if extraModulePath is None or not os.path.exists(extraModulePath):
        return None
    exModule = imp.load_source(MGEAR_EXTRA_ENVIRON, os.path.abspath(extraModulePath))
    additionalFuncDict = getattr(exModule, EXTRA_MODULE_DICT, None)
    if additionalFuncDict is None:
        mc.warning("'{}' not found in {}".format(EXTRA_MODULE_DICT, extraModulePath))
        print("No additional menu items added to {}".format(TOOL_NAME))
    return additionalFuncDict


def selectNode(name):
    """Convenience function, to ensure no errors when selecting nodes in UI

    Args:
        name (str): name of node to be selected
    """
    if mc.objExists(name):
        mc.select(name)
    else:
        print(name, "No longer exists for selection!")


# =============================================================================
# UI General Functions
# =============================================================================


def getControlAttrWidget(nodeAttr, label=""):
    """get a cmds.attrControlGrp wrapped in a qtWidget, still connected
    to the specified attr

    Args:
        nodeAttr (str): node.attr, the target for the attrControlGrp
        label (str, optional): name for the attr widget

    Returns:
        QtWidget: qwidget created from attrControlGrp
    """
    mAttrFeild = mc.attrControlGrp(attribute=nodeAttr, label=label, po=True)
    ptr = mui.MQtUtil.findControl(mAttrFeild)
    if PY2:
        controlWidget = QtCompat.wrapInstance(long(ptr), base=QtWidgets.QWidget)  # type: ignore  # noqa
    else:
        controlWidget = QtCompat.wrapInstance(int(ptr), base=QtWidgets.QWidget)
    controlWidget.setContentsMargins(0, 0, 0, 0)
    controlWidget.setMinimumWidth(0)
    attrEdit = [
        wdgt for wdgt in controlWidget.children() if type(wdgt) == QtWidgets.QLineEdit
    ]
    [
        wdgt.setParent(attrEdit[0])
        for wdgt in controlWidget.children()
        if type(wdgt) == QtCore.QObject
    ]

    attrEdit[0].setParent(None)
    controlWidget.setParent(attrEdit[0])
    controlWidget.setHidden(True)
    return attrEdit[0], mAttrFeild


def HLine():
    """seporator line for widgets

    Returns:
        Qframe: line for seperating UI elements visually
    """
    seperatorLine = QtWidgets.QFrame()
    seperatorLine.setFrameShape(QtWidgets.QFrame.HLine)
    seperatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
    return seperatorLine


def VLine():
    """seporator line for widgets

    Returns:
        Qframe: line for seperating UI elements visually
    """
    seperatorLine = QtWidgets.QFrame()
    seperatorLine.setFrameShape(QtWidgets.QFrame.VLine)
    seperatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
    return seperatorLine


def show(dockable=True, newSceneCallBack=True, *args):
    """To launch the ui and not get the same instance

    Returns:
        DistributeUI: instance

    Args:
        *args: Description
    """
    global RBF_UI
    if "RBF_UI" in globals():
        try:
            RBF_UI.close()  # type: ignore
        except TypeError:
            pass
    RBF_UI = PoseDriverManagerUI(
        parent=pyqt.maya_main_window(), newSceneCallBack=newSceneCallBack
    )
    RBF_UI.show(dockable=True)

    return RBF_UI


def genericWarning(parent, warningText):
    """generic prompt warning with the provided text

    Args:
        parent (QWidget): Qwidget to be parented under
        warningText (str): information to display to the user

    Returns:
        QtCore.Response: of what the user chose. For warnings
    """
    selWarning = QtWidgets.QMessageBox(parent)
    selWarning.setText(warningText)
    results = selWarning.exec_()
    return results


def promptAcceptance(parent, descriptionA, descriptionB):
    """Warn user, asking for permission

    Args:
        parent (QWidget): to be parented under
        descriptionA (str): info
        descriptionB (str): further info

    Returns:
        QtCore.Response: accept, deline, reject
    """
    msgBox = QtWidgets.QMessageBox(parent)
    msgBox.setText(descriptionA)
    msgBox.setInformativeText(descriptionB)
    msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
    msgBox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
    decision = msgBox.exec_()
    return decision


class ClickableLineEdit(QtWidgets.QLineEdit):

    """subclass to allow for clickable lineEdit, as a button

    Attributes:
        clicked (QtCore.Signal): emitted when clicked
    """

    clicked = QtCore.Signal(str)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self.text())
        else:
            super(ClickableLineEdit, self).mousePressEvent(event)


class ClickableListEdit(QtWidgets.QGroupBox):

    """

    Attributes:
        clicked (QtCore.Signal): emitted when clicked
    """

    clicked = QtCore.Signal(str)
    genericWidgetHight = 24

    def __init__(self, parent=None, label="", widgets=None):

        super(ClickableListEdit, self).__init__(parent)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.hLayout = QtWidgets.QHBoxLayout()
        self.vLayout = QtWidgets.QVBoxLayout()
        self.vLayout2 = QtWidgets.QVBoxLayout()

        self.nodeLabel = QtWidgets.QLabel(label)
        self.nodeLabel.setFixedWidth(100)
        self.vLayout.addWidget(self.nodeLabel)

        self.listWidget = QtWidgets.QListWidget(self)
        self.listWidget.setDragDropOverwriteMode(True)
        self.listWidget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.listWidget.setAlternatingRowColors(True)
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidget.setSelectionRectVisible(False)
        self.hLayout.addWidget(self.listWidget)

        self.hLayout.addLayout(self.vLayout2)

        self.addButton = QtWidgets.QPushButton("Add")
        self.addButton.setFixedHeight(self.genericWidgetHight)
        self.vLayout2.addWidget(self.addButton)

        self.removeButton = QtWidgets.QPushButton("Remove")
        self.removeButton.setFixedHeight(self.genericWidgetHight)
        self.vLayout2.addWidget(self.removeButton)

        if widgets is not None:
            if isinstance(widgets, (list, tuple, set)):
                for w in widgets:
                    self.vLayout2.addWidget(w)
            else:
                self.vLayout2.addWidget(widgets)

        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.vLayout2.addItem(spacerItem)
        self.vLayout.addLayout(self.hLayout)

        self.gridLayout.addLayout(self.vLayout, 0, 0, 1, 1)

    def mousePressEvent(self, event):
        # type: (QtGui.QMouseEvent) -> None
        """Event handler to select item in the scene."""

        if event.button() == QtCore.Qt.LeftButton:
            item = self.listWidget.currentItem()
            if item is not None:
                text = item.text()
                self.clicked.emit(text)

        else:
            super(ClickableListEdit, self).mousePressEvent(event)

    def setText(self, items):
        # type: (List[Text]) -> None
        """Set list item"""

        for item in items:
            print(f"{item=}")
            dup = self.listWidget.findItems(item, QtCore.Qt.MatchExactly)
            if not dup:
                self.listWidget.addItems((item,))

    def setItems(self, items):
        # type: (List[Text]) -> None
        """Set list item"""

        for item in items:
            dup = self.listWidget.findItems(item, QtCore.Qt.MatchExactly)
            if not dup:
                self.listWidget.addItems((item,))

    def removeItem(self):
        # type: () -> None
        """Remove current item"""

        row = self.listWidget.currentRow()
        if row == -1:
            row = self.listWidget.count() - 1

        if row < 0:
            return

        self.listWidget.takeItem(row)

    def text(self):
        # type: () -> List[Text]
        """Returns all items as list of text"""
        res = []
        for row in range(self.listWidget.count()):
            res.append(self.listWidget.item(row).text())

        return res

    def clear(self):
        # type: () -> None
        """Clear all entries"""
        return self.listWidget.clear()


class TabBar(QtWidgets.QTabBar):
    """Subclass to get a taller tab widget, for readability"""

    def __init__(self):
        super(TabBar, self).__init__()

    def tabSizeHint(self, index):
        width = QtWidgets.QTabBar.tabSizeHint(self, index).width()
        return QtCore.QSize(width, 25)


class RBFSetupInput(QtWidgets.QDialog):

    """Allow the user to select which attrs will drive the rbf nodes in a setup

    Attributes:
        drivenListWidget (QListWidget): widget to display attrs to drive setup
        okButton (QPushButton): BUTTON
        result (list): of selected attrs from listWidget
        setupField (bool)): Should the setup lineEdit widget be displayed
        setupLineEdit (QLineEdit): name selected by user
    """

    def __init__(self, listValues=(), setupField=True, parent=None):
        """setup the UI widgets

        Args:
            listValues (list): attrs to be displayed on the list
            setupField (bool, optional): should the setup line edit be shown
            parent (QWidget, optional): widget to parent this to
        """
        super(RBFSetupInput, self).__init__(parent=parent)
        self.setWindowTitle(TOOL_TITLE)
        mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(mainLayout)
        self.setupField = setupField
        self.result = None
        #  --------------------------------------------------------------------
        setupLayout = QtWidgets.QHBoxLayout()
        setupLabel = QtWidgets.QLabel("Specify Setup Name")
        self.setupLineEdit = QtWidgets.QLineEdit()
        self.setupLineEdit.setPlaceholderText("<name>_<side><int> // skirt_L0")
        setupLayout.addWidget(setupLabel)
        setupLayout.addWidget(self.setupLineEdit)
        if setupField:
            mainLayout.addLayout(setupLayout)
        #  --------------------------------------------------------------------
        # buttonLayout = QtWidgets.QHBoxLayout()
        self.okButton = QtWidgets.QPushButton("Ok")
        self.okButton.clicked.connect(self.onOK)
        mainLayout.addWidget(self.okButton)

    def onOK(self):
        """collect information from the displayed widgets, userinput, return

        Returns:
            list: of user input provided from user
        """
        setupName = self.setupLineEdit.text()
        if setupName == "" and self.setupField:
            genericWarning(self, "Enter Setup Name")
            return

        self.result = setupName
        self.accept()
        return setupName

    def getValue(self):
        """convenience to get result

        Returns:
            TYPE: Description
        """
        return self.result

    def exec_(self):
        """Convenience

        Returns:
            list: [str, [of selected attrs]]
        """
        super(RBFSetupInput, self).exec_()
        return self.result


class PoseDriverManagerUI(MayaQWidgetDockableMixin, QtWidgets.QMainWindow):

    """A manager for creating, mirroring, importing/exporting poses created
    for RBF type nodes.

    Attributes:
        absWorld (bool): Type of pose info look up, world vs local
        addRbfButton (QPushButton): button for adding RBFs to setup
        allSetupsInfo (dict): setupName:[of all the RBFNodes in scene]
        attrMenu (TYPE): Description
        currentRBFSetupNodes (list): currently selected setup nodes(userSelect)
        driverPoseTableWidget (QTableWidget): poseInfo for the driver node
        genericWidgetHight (int): convenience to adjust height of all buttons
        mousePosition (QPose): if tracking mouse position on UI
        rbfTabWidget (QTabWidget): where the driven table node info is
        displayed
    """

    mousePosition = QtCore.Signal(int, int)

    def __init__(self, parent=None, hideMenuBar=False, newSceneCallBack=True):
        super(PoseDriverManagerUI, self).__init__(parent=parent)
        # UI info -------------------------------------------------------------
        self.callBackID = None
        self.setWindowTitle(TOOL_TITLE)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.genericWidgetHight = 24
        # class info ----------------------------------------------------------
        self.absWorld = True
        self.zeroedDefaults = True
        self.currentRBFSetupNodes = []  # type: List[RBFNode]
        self.allSetupsInfo = {}
        self.setMenuBar(self.createMenuBar(hideMenuBar=hideMenuBar))
        self.setCentralWidget(self.createCentralWidget())
        self.centralWidget().setMouseTracking(True)
        self.refreshRbfSetupList()
        self.connectSignals()
        # added because the dockableMixin makes the ui appear small
        self.adjustSize()
        if newSceneCallBack:
            self.newSceneCallBack()

    def callBackFunc(self, *args):
        """super safe function for trying to refresh the UI, should anything
        fail.

        Args:
            *args: Description
        """
        try:
            self.refresh()
        except Exception:
            pass

    def removeSceneCallback(self):
        """remove the callback associated witht he UI, quietly fail."""
        try:
            om.MSceneMessage.removeCallback(self.callBackID)
        except Exception as e:
            print("CallBack removal failure:")
            print(e)

    def newSceneCallBack(self):
        """create a new scene callback to refresh the UI when scene changes."""
        callBackType = om.MSceneMessage.kSceneUpdate
        try:
            func = self.callBackFunc
            obj = om.MSceneMessage.addCallback(callBackType, func)
            self.callBackID = obj
        except Exception as e:
            print(e)
            self.callBackID = None

    # general functions -------------------------------------------------------
    def getSelectedSetup(self):
        # type: () -> Tuple[Text, Text]
        """return the string name of the selected setup from user and type

        Returns:
            str, str: name, nodeType
        """

        selectedSetup = self.rbf_cbox.currentText()
        if selectedSetup.startswith("New"):
            return None, "poseDriver"

        else:
            return selectedSetup, self.currentRBFSetupNodes[0].rbfType

    def getDrivenNodesFromSetup(self):
        """based on the user selected setup, get the associated RBF nodes

        Returns:
            list: driven rbfnodes
        """
        drivenNodes = []
        for rbfNode in self.currentRBFSetupNodes:
            drivenNodes.extend(rbfNode.getDrivenNode)
        return drivenNodes

    def getUserSetupInfo(self, setupField=True):
        """prompt the user for information needed to create setup or add
        rbf node to existing setup

        Args:
            setupField (bool, optional): should the user be asked to input
            a name for setup

        Returns:
            str: name specified
        """
        userInputWdgt = RBFSetupInput(setupField=setupField, parent=self)
        results = userInputWdgt.exec_()
        if results:
            return results
        else:
            return None

    def __deleteSetup(self):
        decision = promptAcceptance(
            self, "Delete current Setup?", "This will delete all RBF nodes in setup."
        )
        if decision in [QtWidgets.QMessageBox.Discard, QtWidgets.QMessageBox.Cancel]:
            return
        self.deleteSetup()

    def deleteSetup(self, setupName=None):
        """Delete all the nodes within a setup.

        Args:
            setupName (None, optional): Description
        """
        setupType = None
        if setupName is None:
            setupName, setupType = self.getSelectedSetup()
        nodesToDelete = self.allSetupsInfo.get(setupName, [])
        for rbfNode in nodesToDelete:
            drivenNode = rbfNode.getDrivenNode()
            rbfNode.deleteRBFToggleAttr()
            if drivenNode:
                rbf_node.removeDrivenGroup(drivenNode[0])
            mc.delete(rbfNode.transformNode)
        self.refresh()

    def addRBFToSetup(self):
        """query the user in case of a new setup or adding additional RBFs to
        existing.

        Returns:
            TYPE: Description
        """

        # TODO cut this function down to size
        sourceNodes = self.driverLineEdit.text()
        drivenNodes = self.controlLineEdit.text()

        # take every opportunity to return to avoid unneeded processes
        if not sourceNodes or not drivenNodes:
            return

        setupName, rbfType = self.getSelectedSetup()
        if setupName is None:
            setupName = self.getUserSetupInfo()

        # TODO: rotation or translation
        driverAttrs = "rotation"
        # drivenAttrs = "rotation"

        # create RBFNode instance, apply settings
        rbfNode = rbf_manager_ui.sortRBF(
            setupName, rbfType="poseDriver"
        )  # type: RBFNode
        rbfNode.create(sourceNodes, drivenNodes)
        rbfNode.setSetupName(setupName)
        rbfNode.setDriverControlAttr(drivenNodes)
        rbfNode.setDriverNode(sourceNodes, driverAttrs)
        defaultVals = rbfNode.setDrivenNode(drivenNodes)

        # Check if there any preexisting nodes in setup, if so copy pose index
        if self.currentRBFSetupNodes:
            currentRbfs = self.currentRBFSetupNodes[0]
            print("Syncing poses indices from  {} >> {}".format(currentRbfs, rbfNode))
            rbfNode.syncPoseIndices(self.currentRBFSetupNodes[0])

        else:
            rbfNode.addPose()
            self.populateDriverInfo(rbfNode, rbfNode.getNodeInfo())

        # add newly created RBFNode to list of current
        self.currentRBFSetupNodes.append(rbfNode)

        # get info to populate the UI with it
        # weightInfo = rbfNode.getNodeInfo()
        self.refreshRbfSetupList(setToSelection=setupName)
        self.lockDriverWidgets()

        mc.select(drivenNodes)

    def refreshAllTables(self):
        """Convenience function to refresh all the tables on all the tabs
        with latest information.
        """

        weightInfo = None
        rbfNode = None

        for rbfNode in self.currentRBFSetupNodes:
            weightInfo = rbfNode.getNodeInfo()

            print(f"{weightInfo=}, {rbfNode=}")
            if weightInfo and rbfNode:
                self.populateDriverInfo(rbfNode, weightInfo)

    def deletePose(self):
        """delete a pose from the UI and all the RBFNodes in the setup.

        Returns:
            n/a: n/a
        """
        driverRow = self.driverPoseTableWidget.currentRow()
        for rbfNode in self.currentRBFSetupNodes:
            rbfNode.deletePose(indexToPop=driverRow)

        self.refreshAllTables()

    def editPose(self):
        """edit an existing pose. Specify the index

        Returns:
            TYPE: Description
        """
        rbfNodes = self.currentRBFSetupNodes
        if not rbfNodes:
            return

        driverRow = self.driverPoseTableWidget.currentRow()
        driverNodes = rbfNodes[0].getDriverNode()
        driverAttrs = rbfNodes[0].getDriverNodeAttributes()
        poseInputs = rbf_node.getMultipleAttrs(driverNodes, driverAttrs)

        for rbfNode in rbfNodes:
            poseValues = rbfNode.getPoseValues(resetDriven=True)
            rbfNode.addPose(
                poseInput=poseInputs, poseValue=poseValues, posesIndex=driverRow
            )

        self.refreshAllTables()

    def addPose(self):
        """Add pose to rbf nodes in setup. Additional index on all nodes

        Returns:
            TYPE: Description
        """
        rbfNodes = self.currentRBFSetupNodes
        if not rbfNodes:
            return

        driverNode = rbfNodes[0].getDriverNode()[0]
        driverAttrs = rbfNodes[0].getDriverNodeAttributes()
        poseInputs = rbf_node.getMultipleAttrs(driverNode, driverAttrs)

        for rbfNode in rbfNodes:
            poseValues = rbfNode.getPoseValues(
                resetDriven=True, absoluteWorld=self.absWorld
            )
            rbfNode.addPose(poseInput=poseInputs, poseValue=poseValues)

        self.refreshAllTables()

    def updateAllSetupsInfo(self, includeEmpty=False):
        """refresh the instance dictionary of all the setps in the scene.

        Args:
            includeEmpty (bool, optional): there could be rbf nodes with no
            setup names.
        """
        self.allSetupsInfo = {}
        tmp_dict = rbf_node.getRbfSceneSetupsInfo(includeEmpty=includeEmpty)
        for setupName, nodes in tmp_dict.items():
            rbfNodes = [rbf_manager_ui.sortRBF(n, rbfType="poseDriver") for n in nodes]
            self.allSetupsInfo[setupName] = rbfNodes

    def setNodeToField(self, listEdit, multi=True):
        # type: (ClickableLineEdit, bool) -> None
        """take the currently selected node and set its name to the listEdit
        provided

        Args:
            listEdit (ClickableLineEdit): widget to set the name to
            multi (bool, optional): should multiple nodes be supported

        Returns:
            str: str set to the listEdit
        """
        selected = mc.ls(sl=True)
        if not selected:
            return

        if not multi:
            selected = [selected[0]]

        listEdit.setItems(selected)
        # mc.select(cl=True)

        return

    def removeNodeFromField(self, listEdit):
        # type: (ClickableListEdit) -> None
        """Remove from list and update rbf network setups"""
        # TODO: implement later
        listEdit.removeItem()

    def highlightListEntries(self, listWidget, toHighlight):
        """set the items in a listWidget to be highlighted if they are in list

        Args:
            listWidget (QListWidget): list to highlight items on
            toHighlight (list): of things to highlight
        """
        toHighlight = list(toHighlight)
        scrollToItems = []
        for index in range(listWidget.count()):
            # for qt to check for events like keypress
            item = listWidget.item(index)
            itemText = item.text()
            for desired in toHighlight:
                if desired in itemText:
                    item.setSelected(True)
                    scrollToItems.append(item)
                    toHighlight.remove(desired)
        if scrollToItems:
            listWidget.scrollToItem(scrollToItems[0])

    def updateAttributeDisplay(
        self, attrListWidget, driverNames, highlight=[], attrType="all"
    ):
        """update the provided listwidget with the attrs collected from the
        list of nodes provided

        Args:
            attrListWidget (QListWidget): widget to update
            driverNames (list): of nodes to query for attrs to display
            highlight (list, optional): of item entries to highlight
            keyable (bool, optional): should the displayed attrs be keyable

        Returns:
            n/a: n/a
        """
        nodeAttrsToDisplay = []
        if not driverNames:
            return
        elif type(driverNames) != list:
            driverNames = [driverNames]
        nodeAttrsToDisplay = getPlugAttrs(driverNames, attrType=attrType)
        attrListWidget.clear()
        attrListWidget.addItems(sorted(nodeAttrsToDisplay))
        if highlight:
            self.highlightListEntries(attrListWidget, highlight)

    def __deleteAssociatedWidgetsMaya(self, widget, attrName="associatedMaya"):
        """delete core ui items 'associated' with the provided widgets

        Args:
            widget (QWidget): Widget that has the associated attr set
            attrName (str, optional): class attr to query
        """
        if hasattr(widget, attrName):
            for t in getattr(widget, attrName):
                try:
                    mc.deleteUI(t, ctl=True)
                except Exception:
                    pass
        else:
            setattr(widget, attrName, [])

    def __deleteAssociatedWidgets(self, widget, attrName="associated"):
        """delete widget items 'associated' with the provided widgets

        Args:
            widget (QWidget): Widget that has the associated attr set
            attrName (str, optional): class attr to query
        """
        if hasattr(widget, attrName):
            for t in getattr(widget, attrName):
                try:
                    t.deleteLater()
                except Exception:
                    pass
        else:
            setattr(widget, attrName, [])

    def syncDriverTableCells(
        self, attrEdit, rbfAttrPlug, poseIndex, valueIndex, attributeName, *args
    ):
        """When you edit the driver table, it will update all the sibling
        rbf nodes in the setup.

        Args:
            attrEdit (QLineEdit): cell that was edited in the driver table
            rbfAttrPlug (str): node.attr the cell represents
            *args: signal throws additional args
        """
        attr = rbfAttrPlug.partition(".")[2]
        value = attrEdit.text()
        for rbfNode in self.currentRBFSetupNodes:
            attrPlug = "{}.{}".format(rbfNode, attr)
            mc.setAttr(attrPlug, float(value))
            rbfNode.forceEvaluation()

    def setDriverTable(self, rbfNode, weightInfo):
        """Set the driverTable widget with the information from the weightInfo

        Args:
            rbfNode (RBFNode): node to query additional info from
            weightInfo (dict): to pull information from

        Returns:
            n/a: n/a
        """
        poses = weightInfo["poses"]
        # ensure deletion of associated widgets with this parent widget
        self.__deleteAssociatedWidgetsMaya(self.driverPoseTableWidget)
        self.__deleteAssociatedWidgets(self.driverPoseTableWidget)
        self.driverPoseTableWidget.clear()
        columnLen = len(weightInfo["driverAttrs"])
        self.driverPoseTableWidget.setColumnCount(columnLen)
        headerNames = weightInfo["driverAttrs"]
        self.driverPoseTableWidget.setHorizontalHeaderLabels(headerNames)
        poseInputLen = len(poses["poseInput"])
        self.driverPoseTableWidget.setRowCount(poseInputLen)
        if poseInputLen == 0:
            return

        verticalLabels = ["Pose {}".format(index) for index in range(poseInputLen)]
        self.driverPoseTableWidget.setVerticalHeaderLabels(verticalLabels)

        tmpWidgets = []
        mayaUiItems = []
        for rowIndex, poseInput in enumerate(poses["poseInput"]):
            for columnIndex, pValue in enumerate(poseInput):
                # TODO, this is where we get the attrControlGroup
                rbfAttrPlug = "{}.poses[{}].poseInput[{}]".format(
                    rbfNode, rowIndex, columnIndex
                )

                attrEdit, mAttrFeild = getControlAttrWidget(rbfAttrPlug, label="")
                func = partial(
                    self.syncDriverTableCells,
                    attrEdit,
                    rbfAttrPlug,
                    rowIndex,
                    columnIndex,
                    headerNames[columnIndex],
                )
                self.driverPoseTableWidget.setCellWidget(
                    rowIndex, columnIndex, attrEdit
                )
                attrEdit.returnPressed.connect(func)
                tmpWidgets.append(attrEdit)
                mayaUiItems.append(mAttrFeild)

        setattr(self.driverPoseTableWidget, "associated", tmpWidgets)
        setattr(self.driverPoseTableWidget, "associatedMaya", mayaUiItems)

    def lockDriverWidgets(self, lock=True):
        """toggle the ability to edit widgets after they have been set

        Args:
            lock (bool, optional): should it be locked
        """
        self.controlLineEdit.addButton.blockSignals(lock)

    def populateDriverInfo(self, rbfNode, weightInfo):
        """populate the driver widget, driver, control, driving attrs

        Args:
            rbfNode (RBFNode): node for query
            weightInfo (dict): to pull information from, since we have it
        """
        driverNode = weightInfo["driverNode"]
        self.driverLineEdit.setText(driverNode)

        # populate control here
        drivenNode = weightInfo["drivenNode"]
        self.controlLineEdit.setText(drivenNode)

        print(f"populateDriverInfo: {weightInfo=}, {rbfNode=}")
        self.setDriverTable(rbfNode, weightInfo)

    def _associateRBFnodeAndWidget(self, tabDrivenWidget, rbfNode):
        """associates the RBFNode with a widget for convenience when adding,
        deleting, editing

        Args:
            tabDrivenWidget (QWidget): tab widget
            rbfNode (RBFNode): instance to be associated
        """
        setattr(tabDrivenWidget, "rbfNode", rbfNode)

    def setDrivenTable(self, drivenWidget, rbfNode, weightInfo):
        """set the widgets with information from the weightInfo for dispaly

        Args:
            drivenWidget (QWidget): parent widget, the tab to populate
            rbfNode (RBFNode): node associated with widget
            weightInfo (dict): of information to display
        """
        poses = weightInfo["poses"]
        drivenWidget.tableWidget.clear()
        rowCount = len(poses["poseValue"])
        drivenWidget.tableWidget.setRowCount(rowCount)
        drivenAttrs = weightInfo["drivenAttrs"]
        drivenWidget.tableWidget.setColumnCount(len(drivenAttrs))
        drivenWidget.tableWidget.setHorizontalHeaderLabels(drivenAttrs)
        verticalLabels = ["Pose {}".format(index) for index in range(rowCount)]
        drivenWidget.tableWidget.setVerticalHeaderLabels(verticalLabels)
        for rowIndex, poseInput in enumerate(poses["poseValue"]):
            for columnIndex, pValue in enumerate(poseInput):
                rbfAttrPlug = "{}.poses[{}].poseValue[{}]".format(
                    rbfNode, rowIndex, columnIndex
                )
                attrEdit, mAttrFeild = getControlAttrWidget(rbfAttrPlug, label="")
                drivenWidget.tableWidget.setCellWidget(rowIndex, columnIndex, attrEdit)

    def displayRBFSetupInfo(self, index):
        """Display the rbfnodes within the desired setups

        Args:
            index (int): signal information

        """
        rbfSelection = str(self.rbf_cbox.currentText())
        self.refresh(
            rbfSelection=False,
            driverSelection=True,
            drivenSelection=True,
            currentRBFSetupNodes=False,
        )
        if rbfSelection.startswith("New "):
            self.currentRBFSetupNodes = []
            self.lockDriverWidgets(lock=False)
            return

        print(f"{rbfSelection=}")
        rbfNodes = self.allSetupsInfo.get(rbfSelection, [])
        if not rbfNodes:
            return

        print(f"{rbfNodes[0]=}, {type(rbfNodes[0])}")
        self.currentRBFSetupNodes = rbfNodes
        weightInfo = rbfNodes[0].getNodeInfo()
        for k, v in weightInfo.items():
            print(k, v)
        self.populateDriverInfo(rbfNodes[0], weightInfo)
        self.lockDriverWidgets(lock=True)
        # wrapping the following in try due to what I think is a Qt Bug.
        # need to look further into this.
        #   File "rbf_manager_ui.py", line 872, in createAndTagDrivenWidget
        #     header.sectionClicked.connect(self.setConsistentHeaderSelection)
        # AttributeError: 'PySide2.QtWidgets.QListWidgetItem' object has
        # no attribute 'sectionClicked'

    def attrListMenu(self, attributeListWidget, driverLineEdit, QPos, nodeToQuery=None):
        """right click menu for queie qlistwidget

        Args:
            attributeListWidget (QListWidget): widget to display menu over
            driverLineEdit (QLineEdit): widget to query the attrs from
            QPos (QtCore.QPos): due to the signal, used
            nodeToQuery (None, optional): To display attrs from this nodes
            for menu placement

        No Longer Returned:
            n/a: n/a
        """
        if nodeToQuery is None:
            nodeToQuery = str(driverLineEdit.text())
        self.attrMenu = QtWidgets.QMenu()
        parentPosition = attributeListWidget.mapToGlobal(QtCore.QPoint(0, 0))
        menu_item_01 = self.attrMenu.addAction("Display Keyable")
        menu_item_01.setToolTip("Show Keyable Attributes")
        menu_item_01.triggered.connect(
            partial(
                self.updateAttributeDisplay,
                attributeListWidget,
                nodeToQuery,
                attrType="keyable",
            )
        )
        menu2Label = "Display ChannelBox (Non Keyable)"
        menu_item_02 = self.attrMenu.addAction(menu2Label)
        menu2tip = "Show attributes in ChannelBox that are not keyable."
        menu_item_02.setToolTip(menu2tip)
        menu_item_02.triggered.connect(
            partial(
                self.updateAttributeDisplay,
                attributeListWidget,
                nodeToQuery,
                attrType="cb",
            )
        )
        menu_item_03 = self.attrMenu.addAction("Display All")
        menu_item_03.setToolTip("GIVE ME ALL!")
        menu_item_03.triggered.connect(
            partial(
                self.updateAttributeDisplay,
                attributeListWidget,
                nodeToQuery,
                attrType="all",
            )
        )
        self.attrMenu.move(parentPosition + QPos)
        self.attrMenu.show()

    def refreshRbfSetupList(self, setToSelection=False):
        """refresh the list of setups the user may select from

        Args:
            setToSelection (bool, optional): after refresh, set to desired
        """
        self.rbf_cbox.blockSignals(True)
        self.rbf_cbox.clear()
        addNewOfType = [
            "New {} setup".format(rbf) for rbf in rbf_node.SUPPORTED_RBF_NODES
        ]
        self.updateAllSetupsInfo()
        addNewOfType.extend(sorted(self.allSetupsInfo.keys()))
        self.rbf_cbox.addItems(addNewOfType)
        if setToSelection:
            selectionIndex = self.rbf_cbox.findText(setToSelection)
            self.rbf_cbox.setCurrentIndex(selectionIndex)
        else:
            self.lockDriverWidgets(lock=False)
        self.rbf_cbox.blockSignals(False)

    def refresh(
        self,
        rbfSelection=True,
        driverSelection=True,
        drivenSelection=True,
        currentRBFSetupNodes=True,
        *args,
    ):
        """Refreshes the UI

        Args:
            rbfSelection (bool, optional): desired section to refresh
            driverSelection (bool, optional): desired section to refresh
            drivenSelection (bool, optional): desired section to refresh
            currentRBFSetupNodes (bool, optional): desired section to refresh
        """
        if rbfSelection:
            self.refreshRbfSetupList()

        if driverSelection:
            self.controlLineEdit.clear()
            self.driverLineEdit.clear()
            self.__deleteAssociatedWidgetsMaya(self.driverPoseTableWidget)
            self.__deleteAssociatedWidgets(self.driverPoseTableWidget)
            self.driverPoseTableWidget.clear()

        if currentRBFSetupNodes:
            self.currentRBFSetupNodes = []

    def recallDriverPose(self, indexSelected):
        """recall a pose recorded from one of the RBFNodes in currentSelection
        it should not matter when RBFNode in setup is selected as they
        should all be in sync

        Args:
            indexSelected (int): index of the pose to recall

        Returns:
            n/a: nada
        """
        if not self.currentRBFSetupNodes:
            return
        self.currentRBFSetupNodes[0].recallDriverPose(indexSelected)

    def setConsistentHeaderSelection(self, headerIndex):
        """when a pose is selected in one table, ensure the selection in all
        other tables, to avoid visual confusion

        Args:
            headerIndex (int): desired header to highlight
        """
        self.driverPoseTableWidget.blockSignals(True)
        self.driverPoseTableWidget.selectRow(headerIndex)
        self.driverPoseTableWidget.blockSignals(False)

        self.setEditDeletePoseEnabled(enable=True)

    def setEditDeletePoseEnabled(self, enable=False):
        """toggle buttons that can or cannot be selected

        Args:
            enable (bool, optional): to disable vs not
        """
        self.editPoseButton.setEnabled(enable)
        self.deletePoseButton.setEnabled(enable)

    def setDriverControlOnSetup(self):
        """make sure to set the driverControlAttr when the user supplies one

        Args:
            controlName (str): name of the control to set in an attr
        """
        for rbfNode in self.currentRBFSetupNodes:
            rbfNode.setDriverControlAttr(controlName)

    def setSetupDriverControl(self, listEditWidget):
        """should the user wish to set a different driverControl pose setup
        creation, prompt them prior to proceeding

        Args:
            listEditWidget (QLineEdit): to query for the name

        Returns:
            n/a: nada
        """

        if not self.currentRBFSetupNodes:
            self.setNodeToField(listEditWidget)

        elif self.currentRBFSetupNodes:

            textA = "Do you want to change the Control for setup?"
            textB = "This Control that will be used for recalling poses."
            decision = promptAcceptance(self, textA, textB)

            if any(
                (
                    decision == QtWidgets.QMessageBox.Discard,
                    decision == QtWidgets.QMessageBox.Cancel,
                )
            ):
                return

            self.setNodeToField(listEditWidget)
            # self.setDriverControlOnSetup()

    def getRBFNodesInfo(self, rbfNodes):
        """create a dictionary of all the RBFInfo(referred to as
        weightNodeInfo a lot) for export

        Args:
            rbfNodes (list): [of RBFNodes]

        Returns:
            dict: of all the rbfNodes provided
        """
        weightNodeInfo_dict = {}
        for rbf in rbfNodes:
            weightNodeInfo_dict[rbf.name] = rbf.getNodeInfo()

        return weightNodeInfo_dict

    def importNodes(self):
        """import a setup(s) from file select by user

        Returns:
            n/a: nada
        """
        sceneFilePath = mc.file(sn=True, q=True)
        startDir = os.path.dirname(sceneFilePath)
        filePath = rbf_io.fileDialog(startDir, mode=1)
        if filePath is None:
            return

        rbf_io.importRBFs(filePath)
        mc.select(cl=True)
        self.refresh()
        print("RBF setups imported: {}".format(filePath))

    def exportNodes(self, allSetups=True):
        """export all nodes or nodes from current setup

        Args:
            allSetups (bool, optional): If all or setup

        Returns:
            n/a: nada
        """
        # TODO WHEN NEW RBF NODE TYPES ARE ADDED, THIS WILL NEED TO BE RETOOLED
        nodesToExport = []
        if allSetups:
            [nodesToExport.extend(v) for k, v, in self.allSetupsInfo.items()]
        else:
            nodesToExport = self.currentRBFSetupNodes

        nodesToExport = [n.name for n in nodesToExport]
        sceneFilePath = mc.file(sn=True, q=True)
        startDir = os.path.dirname(sceneFilePath)
        filePath = rbf_io.fileDialog(startDir, mode=0)
        if filePath is None:
            return

        exportRBFs(nodesToExport, filePath)

    def gatherMirroredInfo(self, rbfNodes):
        """gather all the info from the provided nodes and string replace
        side information for its mirror. Using mGear standard
        naming convections

        Args:
            rbfNodes (list): [of RBFNodes]

        Returns:
            dict: with all the info mirrored
        """
        mirrorWeightInfo = {}
        for rbfNode in rbfNodes:
            weightInfo = rbfNode.getNodeInfo()
            # connections -----------------------------------------------------
            mrConnections = []
            for pairs in weightInfo["connections"]:
                mrConnections.append(
                    [mString.convertRLName(pairs[0]), mString.convertRLName(pairs[1])]
                )
            weightInfo["connections"] = mrConnections
            # drivenControlName -----------------------------------------------
            mrDrvnCtl = mString.convertRLName(weightInfo["drivenControlName"])
            weightInfo["drivenControlName"] = mrDrvnCtl
            # drivenNode ------------------------------------------------------
            weightInfo["drivenNode"] = [
                mString.convertRLName(n) for n in weightInfo["drivenNode"]
            ]
            # driverControl ---------------------------------------------------
            mrDrvrCtl = mString.convertRLName(weightInfo["driverControl"])
            weightInfo["driverControl"] = mrDrvrCtl
            # driverNode ------------------------------------------------------
            weightInfo["driverNode"] = [
                mString.convertRLName(n) for n in weightInfo["driverNode"]
            ]
            # setupName -------------------------------------------------------
            mrSetupName = mString.convertRLName(weightInfo["setupName"])
            if mrSetupName == weightInfo["setupName"]:
                mrSetupName = "{}{}".format(mrSetupName, MIRROR_SUFFIX)
            weightInfo["setupName"] = mrSetupName
            # transformNode ---------------------------------------------------
            # name
            # parent
            tmp = weightInfo["transformNode"]["name"]
            mrTransformName = mString.convertRLName(tmp)
            weightInfo["transformNode"]["name"] = mrTransformName

            tmp = weightInfo["transformNode"]["parent"]
            if tmp is None:
                mrTransformPar = None
            else:
                mrTransformPar = mString.convertRLName(tmp)
            weightInfo["transformNode"]["parent"] = mrTransformPar
            # name ------------------------------------------------------------
            mirrorWeightInfo[mString.convertRLName(rbfNode.name)] = weightInfo
        return mirrorWeightInfo

    def getMirroredSetupTargetsInfo(self):
        """convenience function to get all the mirrored info for the new side

        Returns:
            dict: mirrored dict information
        """
        setupTargetInfo_dict = {}
        for rbfNode in self.currentRBFSetupNodes:
            mrRbfNode = mString.convertRLName(rbfNode.name)
            mrRbfNode = rbf_manager_ui.sortRBF(mrRbfNode, rbfType="poseDriver")
            drivenNode = rbfNode.getDrivenNode()[0]
            drivenControlNode = rbfNode.getConnectedRBFToggleNode()
            mrDrivenControlNode = mString.convertRLName(drivenControlNode)
            mrDrivenControlNode = pm.PyNode(mrDrivenControlNode)
            setupTargetInfo_dict[pm.PyNode(drivenNode)] = [
                mrDrivenControlNode,
                mrRbfNode,
            ]
        return setupTargetInfo_dict

    def mirrorSetup(self):
        """gather all info on current setup, mirror the info, use the creation
        func from that rbf module type to create the nodes in the setup with
        mirrored information.

        THE ONLY nodes created will be the ones created during normal
        "add pose" creation. Assumption is that all nodes that need drive,
        driven by the setup exist.

        Returns:
            n/a: nada
        """
        if not self.currentRBFSetupNodes:
            return
        aRbfNode = self.currentRBFSetupNodes[0]
        mirrorWeightInfo = self.gatherMirroredInfo(self.currentRBFSetupNodes)
        mrRbfType = aRbfNode.rbfType
        poseIndices = len(aRbfNode.getPoseInfo()["poseInput"])
        rbfModule = rbf_io.RBF_MODULES[mrRbfType]
        rbfModule.createRBFFromInfo(mirrorWeightInfo)
        setupTargetInfo_dict = self.getMirroredSetupTargetsInfo()
        nameSpace = anim_utils.getNamespace(aRbfNode.name)
        mrRbfNodes = [v[1] for k, v in setupTargetInfo_dict.items()]
        [v.setToggleRBFAttr(0) for v in mrRbfNodes]
        mrDriverNode = mrRbfNodes[0].getDriverNode()[0]
        mrDriverAttrs = mrRbfNodes[0].getDriverNodeAttributes()
        driverControl = aRbfNode.getDriverControlAttr()
        driverControl = pm.PyNode(driverControl)
        for index in range(poseIndices):
            aRbfNode.recallDriverPose(index)
            anim_utils.mirrorPose(flip=False, nodes=[driverControl])
            mrData = []
            for srcNode, dstValues in setupTargetInfo_dict.items():
                mrData.extend(anim_utils.calculateMirrorData(srcNode, dstValues[0]))
            for entry in mrData:
                anim_utils.applyMirror(nameSpace, entry)

            poseInputs = rbf_node.getMultipleAttrs(mrDriverNode, mrDriverAttrs)
            for mrRbfNode in mrRbfNodes:
                poseValues = mrRbfNode.getPoseValues(resetDriven=True)
                mrRbfNode.addPose(
                    poseInput=poseInputs, poseValue=poseValues, posesIndex=index
                )
                mrRbfNode.forceEvaluation()
        [v.setToggleRBFAttr(1) for v in mrRbfNodes]
        setupName, rbfType = self.getSelectedSetup()
        self.refreshRbfSetupList(setToSelection=setupName)
        mc.select(cl=True)

    def hideMenuBar(self, x, y):
        """rules to hide/show the menubar when hide is enabled

        Args:
            x (int): coord X of the mouse
            y (int): coord Y of the mouse
        """
        if x < 100 and y < 50:
            self.menuBar().show()
        else:
            self.menuBar().hide()

    def reevalluateAllNodes(self):
        """for evaluation on all nodes in any setup. In case of manual editing"""
        for name, rbfNodes in self.allSetupsInfo.items():
            [rbfNode.forceEvaluation() for rbfNode in rbfNodes]
        print("All Nodes have been Re-evaluated")

    def toggleGetPoseType(self, toggleState):
        """records whether the user wants poses recorded in worldSpace or check
        local space

        Args:
            toggleState (bool): default True
        """
        self.absWorld = toggleState
        print("Recording poses in world space set to: {}".format(toggleState))

    def toggleDefaultType(self, toggleState):
        """records whether the user wants default poses to be zeroed

        Args:
            toggleState (bool): default True
        """
        self.zeroedDefaults = toggleState
        print("Default poses are zeroed: {}".format(toggleState))

    # signal management -------------------------------------------------------
    def connectSignals(self):
        """connect all the signals in the UI
        Exceptions being MenuBar and Table header signals
        """
        self.rbf_cbox.currentIndexChanged.connect(self.displayRBFSetupInfo)

        self.rbf_refreshButton.clicked.connect(self.refresh)

        self.driverLineEdit.clicked.connect(selectNode)
        self.controlLineEdit.clicked.connect(selectNode)
        header = self.driverPoseTableWidget.verticalHeader()
        header.sectionClicked.connect(self.setConsistentHeaderSelection)
        header.sectionClicked.connect(self.recallDriverPose)
        selDelFunc = self.setEditDeletePoseEnabled
        self.driverPoseTableWidget.itemSelectionChanged.connect(selDelFunc)
        self.addRbfButton.clicked.connect(self.addRBFToSetup)

        self.addPoseButton.clicked.connect(self.addPose)
        self.editPoseButton.clicked.connect(self.editPose)
        self.deletePoseButton.clicked.connect(self.deletePose)
        partialObj = partial(self.setSetupDriverControl, self.controlLineEdit)
        self.controlLineEdit.addButton.clicked.connect(partialObj)
        self.driverLineEdit.addButton.clicked.connect(
            partial(self.setNodeToField, self.driverLineEdit)
        )
        self.driverLineEdit.removeButton.clicked.connect(
            partial(self.removeNodeFromField, self.driverLineEdit)
        )
        self.controlLineEdit.removeButton.clicked.connect(
            partial(self.removeNodeFromField, self.controlLineEdit)
        )

    # broken down widgets -----------------------------------------------------
    def createSetupSelectorWidget(self):
        """create the top portion of the weidget, select setup + refresh

        Returns:
            list: QLayout, QCombobox, QPushButton
        """
        setRBFLayout = QtWidgets.QHBoxLayout()
        rbfLabel = QtWidgets.QLabel("Select RBF Setup:")
        rbf_cbox = QtWidgets.QComboBox()
        rbf_refreshButton = QtWidgets.QPushButton("Refresh")
        rbf_cbox.setFixedHeight(self.genericWidgetHight)
        rbf_refreshButton.setMaximumWidth(80)
        rbf_refreshButton.setFixedHeight(self.genericWidgetHight - 1)
        setRBFLayout.addWidget(rbfLabel)
        setRBFLayout.addWidget(rbf_cbox, 1)
        setRBFLayout.addWidget(rbf_refreshButton)
        return setRBFLayout, rbf_cbox, rbf_refreshButton

    def selectNodeWidget(self, label, widgets):
        """create a lout with label, lineEdit, QPushbutton for user input"""
        nodeLayout = QtWidgets.QHBoxLayout()
        listEdit = ClickableListEdit(label=label, widgets=widgets)
        nodeLayout.addWidget(listEdit)

        return nodeLayout, listEdit

    def labelListWidget(self, label, horizontal=True):
        """create the listAttribute that users can select their driver/driven
        attributes for the setup

        Args:
            label (str): to display above the listWidget
            horizontal (bool, optional): should the label be above or infront
            of the listWidget

        Returns:
            list: QLayout, QListWidget
        """
        if horizontal:
            attributeLayout = QtWidgets.QHBoxLayout()
        else:
            attributeLayout = QtWidgets.QVBoxLayout()
        attributeLabel = QtWidgets.QLabel(label)
        attributeListWidget = QtWidgets.QListWidget()
        attributeLayout.addWidget(attributeLabel)
        attributeLayout.addWidget(attributeListWidget)
        return attributeLayout, attributeListWidget

    def addRemoveButtonWidget(self, label1, label2, horizontal=True):
        if horizontal:
            addRemoveLayout = QtWidgets.QHBoxLayout()
        else:
            addRemoveLayout = QtWidgets.QVBoxLayout()
        addAttributesButton = QtWidgets.QPushButton(label1)
        removeAttributesButton = QtWidgets.QPushButton(label2)
        addRemoveLayout.addWidget(addAttributesButton)
        addRemoveLayout.addWidget(removeAttributesButton)
        return addRemoveLayout, addAttributesButton, removeAttributesButton

    def createDriverAttributeWidget(self):
        # type () -> Tuple[ClickableLineEdit, ClickableLineEdit, QtWidget.QVBoxLayout, QtWidgets.QComboBox, QtWidgets.QComboBox]
        """widget where the user inputs information for the setups

        Returns:
            tuple: [of widgets]
        """
        driverMainLayout = QtWidgets.QVBoxLayout()

        #  --------------------------------------------------------------------
        source = QtWidgets.QComboBox()
        source.addItems(("rotation", "translation"))
        axis = QtWidgets.QComboBox()
        axis.addItems(("x", "y", "z"))

        (driverLayout, driverLineEdit) = self.selectNodeWidget("Source Bones", (source, axis))
        driverLineEdit.setToolTip("The node driving the setup. (Click me!)")

        #  --------------------------------------------------------------------
        (drivenLayout, drivenLineEdit) = self.selectNodeWidget("Drive Bones", None)
        drivenLineEdit.setToolTip("The node driving the setup. (Click me!)")

        #  --------------------------------------------------------------------
        driverMainLayout.addLayout(driverLayout, 0)
        driverMainLayout.addLayout(drivenLayout, 0)

        return (drivenLineEdit, driverLineEdit, driverMainLayout, axis, source)

    def createTableWidget(self):
        """create table widget used to display poses, set tooltips and colum

        Returns:
            QTableWidget: QTableWidget
        """
        tableWidget = QtWidgets.QTableWidget()
        tableWidget.insertColumn(0)
        tableWidget.insertRow(0)
        tableWidget.setHorizontalHeaderLabels(["Pose Value"])
        tableWidget.setVerticalHeaderLabels(["Pose #0"])
        tableTip = "Live connections to the RBF Node in your setup."
        tableTip = tableTip + "\nSelect the desired Pose # to recall pose."
        tableWidget.setToolTip(tableTip)
        return tableWidget

    def createTabWidget(self):
        """Tab widget to add driven widgets too. Custom TabBar so the tab is
        easier to select

        Returns:
            QTabWidget:
        """
        tabLayout = QtWidgets.QTabWidget()
        tabLayout.setContentsMargins(0, 0, 0, 0)
        tabBar = TabBar()
        tabLayout.setTabBar(tabBar)
        tabBar.setTabsClosable(True)
        return tabLayout

    def createOptionsButtonsWidget(self):
        """add, edit, delete buttons for modifying rbf setups.

        Returns:
            list: [QPushButtons]
        """
        optionsLayout = QtWidgets.QHBoxLayout()
        addPoseButton = QtWidgets.QPushButton("Add Pose")
        addTip = "After positioning all controls in the setup, add new pose."
        addTip = addTip + "\nEnsure the driver node has a unique position."
        addPoseButton.setToolTip(addTip)
        addPoseButton.setFixedHeight(self.genericWidgetHight)
        EditPoseButton = QtWidgets.QPushButton("Edit Pose")
        EditPoseButton.setToolTip("Recall pose, adjust controls and Edit.")
        EditPoseButton.setFixedHeight(self.genericWidgetHight)
        deletePoseButton = QtWidgets.QPushButton("Delete Pose")
        deletePoseButton.setToolTip("Recall pose, then Delete")
        deletePoseButton.setFixedHeight(self.genericWidgetHight)
        optionsLayout.addWidget(addPoseButton)
        optionsLayout.addWidget(EditPoseButton)
        optionsLayout.addWidget(deletePoseButton)
        return (optionsLayout, addPoseButton, EditPoseButton, deletePoseButton)

    def createMenuBar(self, hideMenuBar=False):
        """Create the UI menubar, with option to hide based on mouse input

        Args:
            hideMenuBar (bool, optional): should it autoHide

        Returns:
            QMenuBar: for parenting
        """
        mainMenuBar = QtWidgets.QMenuBar()
        mainMenuBar.setContentsMargins(0, 0, 0, 0)
        file = mainMenuBar.addMenu("File")
        menu1 = file.addAction("Re-evaluate Nodes", self.reevalluateAllNodes)
        menu1.setToolTip("Force all RBF nodes to re-revaluate.")
        file.addAction("Export All", self.exportNodes)
        file.addAction(
            "Export current setup", partial(self.exportNodes, allSetups=False)
        )
        file.addAction("Import RBFs", partial(self.importNodes))
        file.addSeparator()
        file.addAction("Delete Current Setup", self.__deleteSetup)
        # mirror --------------------------------------------------------------
        mirrorMenu = mainMenuBar.addMenu("Mirror")
        mirrorMenu1 = mirrorMenu.addAction("Mirror Setup", self.mirrorSetup)
        mirrorMenu1.setToolTip("This will create a new setup.")

        # settings ------------------------------------------------------------
        settingsMenu = mainMenuBar.addMenu("Settings")
        menuLabel = "Add poses in worldSpace"
        worldSpaceMenuItem = settingsMenu.addAction(menuLabel)
        worldSpaceMenuItem.toggled.connect(self.toggleGetPoseType)

        worldSpaceMenuItem.setCheckable(True)
        worldSpaceMenuItem.setChecked(True)
        toolTip = "When ADDING NEW pose, should it be recorded in worldSpace."

        menuLabel = "Default Poses is Zeroed"
        zeroedDefaultsMenuItem = settingsMenu.addAction(menuLabel)
        zeroedDefaultsMenuItem.toggled.connect(self.toggleDefaultType)

        zeroedDefaultsMenuItem.setCheckable(True)
        zeroedDefaultsMenuItem.setChecked(True)

        worldSpaceMenuItem.setToolTip(toolTip)

        # show override -------------------------------------------------------
        additionalFuncDict = getEnvironModules()
        if additionalFuncDict:
            showOverridesMenu = mainMenuBar.addMenu("Local Overrides")
            for k, v in additionalFuncDict.items():
                showOverridesMenu.addAction(k, v)

        if hideMenuBar:
            mainMenuBar.hide()
            self.setMouseTracking(True)
            self.mousePosition.connect(self.hideMenuBar)

        return mainMenuBar

    # main assebly ------------------------------------------------------------

    def createCentralWidget(self):
        """main UI assembly

        Returns:
            QtWidget: main UI to be parented to as the centralWidget
        """
        centralWidget = QtWidgets.QWidget()
        centralWidgetLayout = QtWidgets.QVBoxLayout()
        centralWidget.setLayout(centralWidgetLayout)
        (
            rbfLayout,
            self.rbf_cbox,
            self.rbf_refreshButton,
        ) = self.createSetupSelectorWidget()
        self.rbf_cbox.setToolTip("List of available setups in the scene.")
        self.rbf_refreshButton.setToolTip("Refresh the UI")
        centralWidgetLayout.addLayout(rbfLayout)
        centralWidgetLayout.addWidget(HLine())
        #  --------------------------------------------------------------------
        driverDrivenLayout = QtWidgets.QHBoxLayout()
        (
            self.controlLineEdit,
            self.driverLineEdit,
            driverLayout,
            self.driverAxisCBox,
            self.driverSourceAttrCBox,
        ) = self.createDriverAttributeWidget()

        self.addRbfButton = QtWidgets.QPushButton("New PoseDriver")
        self.addRbfButton.setToolTip("Select node to be driven by setup.")
        self.addRbfButton.setFixedHeight(self.genericWidgetHight)
        self.addRbfButton.setStyleSheet("background-color: rgb(23, 158, 131)")
        driverLayout.addWidget(self.addRbfButton)

        self.driverPoseTableWidget = self.createTableWidget()
        driverDrivenLayout.addLayout(driverLayout, 0)
        driverDrivenLayout.addWidget(self.driverPoseTableWidget, 1)
        centralWidgetLayout.addLayout(driverDrivenLayout, 1)
        #  --------------------------------------------------------------------
        (
            optionsLayout,
            self.addPoseButton,
            self.editPoseButton,
            self.deletePoseButton,
        ) = self.createOptionsButtonsWidget()
        self.editPoseButton.setEnabled(False)
        self.deletePoseButton.setEnabled(False)
        centralWidgetLayout.addWidget(HLine())
        centralWidgetLayout.addLayout(optionsLayout)
        return centralWidget

    # overrides ---------------------------------------------------------------
    def mouseMoveEvent(self, event):
        """used for tracking the mouse position over the UI, in this case for
        menu hiding/show

        Args:
            event (Qt.QEvent): events to filter
        """
        if event.type() == QtCore.QEvent.MouseMove:
            if event.buttons() == QtCore.Qt.NoButton:
                pos = event.pos()
                self.mousePosition.emit(pos.x(), pos.y())

    def closeEvent(self, evnt):
        """on UI close, ensure that all attrControlgrps are destroyed in case
        the user is just reopening the UI. Properly severs ties to the attrs

        Args:
            evnt (Qt.QEvent): Close event called
        """
        self.__deleteAssociatedWidgetsMaya(self.driverPoseTableWidget)
        self.__deleteAssociatedWidgets(self.driverPoseTableWidget)
        if self.callBackID is not None:
            self.removeSceneCallback()
        super(PoseDriverManagerUI, self).closeEvent(evnt)


if __name__ == "__main__":
    import mgear.rigbits.posedriver_ui as ui

    ui.show()
