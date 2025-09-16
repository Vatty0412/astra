import {} from "express-validator";

export const userValSchema = {
    mobile: {
        notEmpty: {
            errorMessage: "Mobile number is required",
        },
        isLength: {
            options: { min: 10, max: 10 },
            errorMessage: "Mobile number must be 10 digits long",
        },
        matches: {
            options: [/^\+[1-9]\d{1,14}$/], 
            errorMessage: "Mobile number must start with '+' followed by country code and digits (E.164 format)",
        },
        trim: true,
    },
    password: {
        notEmpty: {
            errorMessage: "Password is required"
        },
        isLength: {
            options: { min: 6 },
            errorMessage: "Password must be at least 6 characters long"
        },
    }
};