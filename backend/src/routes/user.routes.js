import {  Router } from "express";
import { checkSchema } from "express-validator";
import { userValSchema } from "../models/validationSchema.js";
import { checkForUser, verifyOTP } from "../controllers/user.controller.js";

const router = Router();

// Define your user-related routes here
router.get("/", (_, res) => { res.send("User route"); });

router.post("/register/check-database", checkSchema(userValSchema), checkForUser);
router.post("/register/verify-otp", verifyOTP);

export default router;