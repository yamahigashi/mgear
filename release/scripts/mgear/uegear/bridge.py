#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
ueGear Client to interact with ueGear Commands within Unreal Engine.
"""

from __future__ import print_function, division, absolute_import

import os
import json
import pprint
import tempfile
import traceback
from urllib.request import urlopen, Request

import pymel.core as pm
import maya.cmds as cmds

from mgear.core import pyFBX
from mgear.uegear import log, utils, tag, io

logger = log.uegear_logger


class UeGearBridge(object):
    """
    Unreal Engine Gear Bridge
    """

    def __init__(self, port=30010, host_address='127.0.0.1'):
        super(UeGearBridge, self).__init__()

        self._host_address = host_address
        self._port = port
        self._timeout = 1000                    # connection to the server will time out after this value.
        self._echo_execution = True             # whether client should print the response coming from server.
        self._echo_payload = True               # whether client should print the JSON payload it's sending to server.
        self._is_executing = False              # whether client is still executing a command.
        self._commands_object_path = '/Engine/PythonTypes.Default__PyUeGearCommands'
        self._headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    # =================================================================================================================
    # PROPERTIES
    # =================================================================================================================

    @property
    def port(self):
        return self._port

    @property
    def host_address(self):
        return self._host_address

    @property
    def is_executing(self):
        return self._is_executing

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = value

    @property
    def echo_execution(self):
        return self._echo_execution

    @echo_execution.setter
    def echo_execution(self, value):
        self._echo_execution = value

    @property
    def echo_payload(self):
        return self._echo_payload

    @echo_payload.setter
    def echo_payload(self, value):
        self._echo_payload = value

    @property
    def commands_object_path(self):
        return self._commands_object_path

    @commands_object_path.setter
    def commands_object_path(self, value):
        self._commands_object_path = value

    # =================================================================================================================
    # BASE
    # =================================================================================================================

    def execute(self, command, parameters=None, timeout=0):
        """
        Executes given command for this client. The server will look for this command in the modules it has loaded.

        :param str command:  The command name that you want to execute within PyUeGearCommands class.
        :param dict parameters: arguments for the command to execute. These have to match the argument names on the
            function exactly.
        :param float timeout: time in seconds after which the request will timeout.
        :return: response coming from the Unreal Remote Server.
        :rtype: dict
        """

        self._is_executing = True
        timeout = timeout if timeout > 0 else self._timeout
        parameters = parameters or dict()

        url = 'http://{}:{}/remote/object/call'.format(self._host_address, self._port)
        payload = {
            'objectPath': self._commands_object_path,
            'functionName': command,
            'parameters': parameters,
            'generateTransaction': True
        }
        try:
            request = Request(url, json.dumps(payload).encode('ascii'), self._headers, method='PUT')
            with urlopen(request, timeout=timeout) as response:
                response = json.load(response)
        except Exception:
            response = {'return': False}
        try:
            evaluated_return = eval(response.get('return'))
            response = {'return': evaluated_return}
        except Exception:
            pass

        if self._echo_payload:
            pprint.pprint(payload)

        if self._echo_execution:
            pprint.pprint(response)

        self._is_executing = False

        return response

    # ==================================================================================================================
    # ASSETS
    # ==================================================================================================================

    def import_selected_assets_from_unreal(self):
        """
        Import current selected Content Browser Unreal assets into the Maya scene.
        """

        temp_folder = tempfile.gettempdir()
        asset_export_datas = self.execute(
            'export_selected_assets', parameters={'directory': temp_folder}).get('ReturnValue', list())

        for asset_export_data in asset_export_datas:
            fbx_file = asset_export_data.get('fbx_file', None)
            if not fbx_file or not os.path.isfile(fbx_file):
                continue
            imported_nodes = utils.import_fbx(fbx_file)
            transform_nodes = cmds.ls(imported_nodes, type='transform')
            for transform_node in transform_nodes:
                asset_type = asset_export_data.get('asset_type', '')
                asset_name = asset_export_data.get('name', '')
                asset_path = asset_export_data.get('path', '')
                if asset_type:
                    tag.apply_tag(transform_node, tag.TAG_ASSET_TYPE_ATTR_NAME, asset_type)
                else:
                    tag.auto_tag(transform_node)
                if asset_name:
                    tag.apply_tag(transform_node, tag.TAG_ASSET_NAME_ATTR_NAME, asset_name)
                if asset_path:
                    tag.apply_tag(transform_node, tag.TAG_ASSET_PATH_ATTR_NAME, asset_path)

    # ==================================================================================================================
    # TRANSFORM COMMANDS
    # ==================================================================================================================

    def update_selected_transforms(self):
        """
        Updates matching Unreal objects within current level with the transforms of the currently selected
        objects within Maya scene.
        """

        selected_nodes = pm.selected()
        old_rotation_orders = list()
        for selected_node in selected_nodes:
            old_rotation_orders.append(selected_node.getRotationOrder())
            selected_node.setRotationOrder('XZY', True)
        try:
            objects = cmds.ls(sl=True, sn=True)
            for obj in objects:
                ue_world_transform = utils.get_unreal_engine_transform_for_maya_node(obj)
                result = self.execute('set_actor_world_transform', parameters={
                    'actor_name': obj,
                    'translation': str(ue_world_transform['rotatePivot']),
                    'rotation': str(ue_world_transform['rotation']),
                    'scale': str(ue_world_transform['scale']),
                })
        finally:
            for i, selected_node in enumerate(selected_nodes):
                selected_node.setRotationOrder(old_rotation_orders[i], True)

    # ==================================================================================================================
    # STATIC MESHES
    # ==================================================================================================================

    def update_static_mesh(self, export_options=None):

        default_export_options = {
            'GenerateLog': False,
            'AnimationOnly': False,
            'Shapes': True,
            'Skins': False,
            'SmoothingGroups': True
        }
        export_options = export_options or dict()
        default_export_options.update(export_options)

        selected_mesh = utils.get_first_in_list(pm.ls(sl=True, type='transform'))
        if not selected_mesh:
            logger.warning('No selected mesh to export')
            return

        meshes = selected_mesh.getShapes() if selected_mesh else None
        if not meshes:
            logger.warning('Selected node has no shapes to export')
            return

        temp_folder = tempfile.gettempdir()
        fbx_temp_file_path = os.path.join(temp_folder, 'uegear_temp_static_mesh.fbx')
        try:
            utils.touch_path(fbx_temp_file_path)
            fbx_export_path = os.path.normpath(fbx_temp_file_path).replace('\\', '/')
            pyFBX.FBXExportGenerateLog(v=default_export_options.get('GenerateLog', False))
            pyFBX.FBXExportAnimationOnly(v=default_export_options.get('AnimationOnly', False))
            pyFBX.FBXExportShapes(v=default_export_options.get('Shapes', True))
            pyFBX.FBXExportSkins(v=default_export_options.get('Skins', False))
            pyFBX.FBXExportSmoothingGroups(v=default_export_options.get('SmoothingGroups', True))
            pyFBX.FBXExport(f=fbx_export_path, s=True)
        except Exception as exc:
            logger.error('Something went wrong while exporting static mesh: {}'.format(traceback.format_exc()))
        finally:

            try:
                pass
            except Exception as exc:
                logger.error('Something went wrong while importing static mesh into Unreal: {}'.format(
                    traceback.format_exc()))

            try:
                utils.get_permission(fbx_temp_file_path)
            except Exception:
                pass
            try:
                os.remove(fbx_temp_file_path)
            except Exception:
                pass

    # ==================================================================================================================
    # LAYOUT COMMANDS
    # ==================================================================================================================

    def import_layout_from_unreal(self):

        temp_folder = tempfile.gettempdir()
        temp_assets_folder = utils.clean_path(os.path.join(temp_folder, 'uegear_temp_assets'))
        if os.path.isdir(temp_assets_folder):
            utils.safe_delete_folder(temp_assets_folder)
        utils.ensure_folder_exists(temp_assets_folder)
        result = self.execute('export_maya_layout', parameters={'directory': temp_assets_folder}).get('ReturnValue', '')
        if result and os.path.isfile(result):
            layout_data = utils.read_json_file(result)
            if layout_data:
                for actor_data in layout_data:
                    fbx_file = actor_data.get('assetExportPath', None)
                    if not fbx_file or not os.path.isfile(fbx_file):
                        continue
                    imported_nodes = utils.import_fbx(fbx_file)
                    transform_nodes = cmds.ls(imported_nodes, type='transform')
                    transform_node = utils.get_first_in_list(transform_nodes)
                    if not transform_node:
                        continue
                    for transform_node in transform_nodes:
                        asset_guid = actor_data.get('guid', '')
                        asset_type = actor_data.get('assetType', '')
                        asset_name = actor_data.get('assetName', '')
                        asset_path = actor_data.get('assetPath', '')
                        actor_name = actor_data['name']
                        translation = actor_data['translation']
                        rotation = actor_data['rotation']
                        scale = actor_data['scale']
                        tag.apply_tag(transform_node, tag.TAG_ASSET_GUID_ATTR_NAME, asset_guid)
                        if asset_type:
                            tag.apply_tag(transform_node, tag.TAG_ASSET_TYPE_ATTR_NAME, asset_type)
                        else:
                            tag.auto_tag(transform_node)
                        tag.apply_tag(transform_node, tag.TAG_ASSET_NAME_ATTR_NAME, asset_name)
                        tag.apply_tag(transform_node, tag.TAG_ASSET_PATH_ATTR_NAME, asset_path)
                        tag.apply_tag(transform_node, tag.TAG_ACTOR_NAME_ATTR_NAME, actor_name)
                    transform_node = cmds.rename(transform_node, actor_name)
                    cmds.setAttr(transform_node + '.translateX', translation[0])
                    cmds.setAttr(transform_node + '.translateY', translation[2])
                    cmds.setAttr(transform_node + '.translateZ', translation[1])
                    cmds.rotate(rotation[0], -rotation[2], rotation[1] * -1, transform_node, r=True)
                    # cmds.setAttr(transform_node + '.rotateX', rotation[0])
                    # cmds.setAttr(transform_node + '.rotateY', rotation[2])
                    # cmds.setAttr(transform_node + '.rotateZ', rotation[1]*-1)
                    cmds.setAttr(transform_node + '.scaleX', scale[0])
                    cmds.setAttr(transform_node + '.scaleY', scale[2])
                    cmds.setAttr(transform_node + '.scaleZ', scale[1])

        utils.safe_delete_folder(temp_assets_folder)

        return True

    def export_layout(self, nodes=None):

        nodes = utils.force_list(nodes or cmds.ls(sl=True))
        if not nodes:
            logger.warning('No layout nodes selected to export')
            return False

        temp_folder = tempfile.gettempdir()
        layout_file = io.export_layout_json(nodes=nodes, output_path=temp_folder)
        result = self.execute('import_maya_layout_from_file', parameters={'layout_file': layout_file})
        utils.safe_delete_file(layout_file)

        return True

    # ==================================================================================================================
    # CAMERA COMMANDS
    # ==================================================================================================================

    def export_cameras(self, cameras=None):

        cameras = utils.force_list(cameras or utils.get_selected_cameras())
        if not cameras:
            logger.warning('No cameras to export')
            return False

        temp_folder = tempfile.gettempdir()
        cameras_file = io.export_layout_json(nodes=cameras, output_path=temp_folder)
        result = self.execute('import_maya_data_from_file', parameters={'data_file': cameras_file})
        utils.safe_delete_file(cameras_file)

        return True
