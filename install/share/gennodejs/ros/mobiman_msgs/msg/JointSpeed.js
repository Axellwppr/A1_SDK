// Auto-generated. Do not edit!

// (in-package mobiman_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class JointSpeed {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.speed = null;
    }
    else {
      if (initObj.hasOwnProperty('speed')) {
        this.speed = initObj.speed
      }
      else {
        this.speed = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type JointSpeed
    // Serialize message field [speed]
    bufferOffset = _arraySerializer.float32(obj.speed, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type JointSpeed
    let len;
    let data = new JointSpeed(null);
    // Deserialize message field [speed]
    data.speed = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.speed.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'mobiman_msgs/JointSpeed';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'c2d600321673904df95b6b086e7cca22';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32[] speed
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new JointSpeed(null);
    if (msg.speed !== undefined) {
      resolved.speed = msg.speed;
    }
    else {
      resolved.speed = []
    }

    return resolved;
    }
};

module.exports = JointSpeed;
