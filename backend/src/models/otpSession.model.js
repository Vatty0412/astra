import mongoose, { Schema } from "mongoose";

const otpSessionSchema = new Schema({
    userId: {
        type: mongoose.Schema.Types.String,
        required: true,
        unique: true
    },
    otp: {
        type: mongoose.Schema.Types.Number,
        required: true
    },
    createdAt: {
        type: mongoose.Schema.Types.Date,
        default: Date.now(),
        expires: 5 * 60
    }
});

export const otpSession = mongoose.model("otpSession", otpSessionSchema);