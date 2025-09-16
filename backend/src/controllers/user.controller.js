import { User } from "./user.model.js";
import { otpSession } from "../models/otpSession.model.js";
import { sendMessage as sendOTP } from "../utils/twilio.util.js";
import { validationResult } from "express-validator";

export const checkForUser = async (req, res) => {
    // Server-side validation
    const results = validationResult(req);
    if(!results.isEmpty()) return res.status(400).json({ message: results.array() });
    
    // Check if user with same mobile number exists
    const { user: { name, mobile, password } } = req;
    const retrievedUser = await User.findOne({ mobile });
    if(retrievedUser && retrievedUser.status) return res.status(409).json({ message: "User by the same mobile number is already registered." });

    // Create a new user in DB with OTP verified status set to false
    const newUser = await User.create({ name, mobile, password });
    if(!newUser) return res.status(500).json({ message: "Internal server error. Please try again later." });

    // Send OTP to user's mobile
    const otp = Math.floor(10 ** 5 + Math.random() * 9 * 10 ** 5);
    try {

        await otpSession.create({ userId: newUser._id, otp });
        await sendOTP(mobile, otp);

    } catch (error) {
        return res.status(500).json({ message: "Could not send OTP. Please try again later." });
    }


    // Set a cookie with registration token
    res
        .status(200)
        .json({ message: "OTP sent successfully." })
        .cookie("registrationToken", newUser._id, {
            httpOnly: true,
            signed: true,
            maxAge: 5 * 60 * 1000, // 5 minutes
        }); 
}

export const verifyOTP = async (req, res) => {
    const { otp } = req.body;
    const { registrationToken } = req.signedCookies;
    if(!registrationToken) return res.status(401).json({ message: "Unauthorized. No registration token found." });

    const session = await otpSession.findOne({ userId: registrationToken });
    if(!session) return res.status(400).json({ message: "OTP session expired. Please register again." });

    if(session.otp !== parseInt(otp)) return res.status(400).json({ message: "Invalid OTP. Please try again." });
    await User.updateOne({ _id: registrationToken }, { otpVerified: true });
    await otpSession.deleteOne({ userId: registrationToken });

    res
        .status(200)
        .clearCookie("registrationToken")
        .json({ message: "OTP verified successfully. User registered." });
}