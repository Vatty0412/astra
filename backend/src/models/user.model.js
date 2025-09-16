import mongoose, { Schema } from "mongoose";

const userSchema = new Schema({
    name: {
        type: Schema.Types.String,
        required: true,
    },
    mobile: {
        type: Schema.Types.String,
        required: true,
        unique: true,
    },
    password: {
        type: Schema.Types.String,
        required: true,
    },
    otpVerified: {
        type: Schema.Types.Boolean,
        default: false,
    }
}, { timestamps: true });

export const User = mongoose.model("User", userSchema);