"""
src/rules.py

Rule-based policy evaluation engine for the Singapore Silver Support Scheme (2025/2026 guidelines).
Handles mathematical calculations, income criteria, housing checks, and quarterly payout determinations.
"""

from typing import Dict, Any, List


def evaluate_silver_support_eligibility(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates Silver Support Scheme eligibility and quarterly payout based on 2025/2026 guidelines.

    Args:
        user_data (dict): Input parameter dictionary containing:
            - age (int): Age of applicant.
            - total_cpf_contributions_at_55 (float): Total CPF contributions at age 55.
            - is_self_employed_or_platform (bool): True if applicant is self-employed or platform worker.
            - net_trade_income_avg (float): Average annual Net Trade Income (if self-employed).
            - hdb_flat_type (str): '1-2 Room', '3-Room', '4-Room', '5-Room Live-In', or '5-Room Owned / Private'.
            - owns_private_or_multiple_properties (bool): Applicant property ownership flag.
            - spouse_owns_private_or_multiple_properties (bool): Spouse property ownership flag.
            - monthly_household_income (float): Gross monthly household income.
            - household_members_count (int): Total household size.
            - on_comcare_lta (bool): True if receiving ComCare Long-Term Assistance.

    Returns:
        dict: Evaluation output containing:
            - is_eligible (bool)
            - payout_amount (float): Quarterly payout in SGD ($0.0 if ineligible).
            - pchhi (float): Per Capita Household Income.
            - passed_criteria (List[str]): List of met criteria descriptions.
            - failed_criteria (List[str]): List of failed criteria descriptions.
    """
    # 1. Extract and normalize inputs
    age: int = int(user_data.get("age", 0))
    total_cpf: float = float(user_data.get("total_cpf_contributions_at_55", 0.0))
    is_self_employed: bool = bool(user_data.get("is_self_employed_or_platform", False))
    net_trade_income: float = float(user_data.get("net_trade_income_avg", 0.0))

    flat_type: str = str(user_data.get("hdb_flat_type", "")).strip()
    owns_pvt: bool = bool(user_data.get("owns_private_or_multiple_properties", False))
    spouse_owns_pvt: bool = bool(user_data.get("spouse_owns_private_or_multiple_properties", False))

    monthly_income: float = float(user_data.get("monthly_household_income", 0.0))
    members_count: int = max(1, int(user_data.get("household_members_count", 1)))
    on_comcare_lta: bool = bool(user_data.get("on_comcare_lta", False))

    # 2. Compute Per Capita Household Income (PCHHI)
    pchhi: float = round(monthly_income / members_count, 2)

    passed_criteria: List[str] = []
    failed_criteria: List[str] = []

    # -------------------------------------------------------------
    # Special Rule: ComCare LTA Override
    # -------------------------------------------------------------
    if on_comcare_lta:
        if age >= 65:
            passed_criteria.extend([
                f"Age criterion met: Applicant is {age} years old (>= 65).",
                "ComCare Long-Term Assistance (LTA) override applied: Automatically eligible for flat $430/quarter payout."
            ])
            return {
                "is_eligible": True,
                "payout_amount": 430.0,
                "pchhi": pchhi,
                "passed_criteria": passed_criteria,
                "failed_criteria": []
            }
        else:
            failed_criteria.append(
                f"ComCare LTA override requires applicant to be at least 65 years old (current age: {age})."
            )

    # -------------------------------------------------------------
    # Standard Criteria Evaluation
    # -------------------------------------------------------------

    # Rule 1: Age Check (>= 65)
    if age >= 65:
        passed_criteria.append(f"Age criterion met: {age} years old (>= 65).")
    else:
        failed_criteria.append(f"Age criterion failed: Applicant is {age} years old (must be at least 65).")

    # Rule 2: CPF / Net Trade Income Check
    if is_self_employed:
        if net_trade_income <= 27600.0:
            passed_criteria.append(
                f"Income criterion met (Self-Employed/Platform): Average Net Trade Income of ${net_trade_income:,.2f} is <= $27,600."
            )
        else:
            failed_criteria.append(
                f"Income criterion failed (Self-Employed/Platform): Average Net Trade Income of ${net_trade_income:,.2f} exceeds $27,600 limit."
            )
    else:
        if total_cpf <= 140000.0:
            passed_criteria.append(
                f"CPF contribution criterion met: Total CPF at age 55 of ${total_cpf:,.2f} is <= $140,000."
            )
        else:
            failed_criteria.append(
                f"CPF contribution criterion failed: Total CPF at age 55 of ${total_cpf:,.2f} exceeds $140,000 limit."
            )

    # Rule 3: Property Ownership Check
    is_5room_owned_or_private = flat_type.casefold() == "5-room owned / private".casefold()
    if owns_pvt or spouse_owns_pvt or is_5room_owned_or_private:
        reasons = []
        if is_5room_owned_or_private:
            reasons.append("Resides in/owns a 5-Room flat or private property")
        if owns_pvt:
            reasons.append("Applicant owns private property or multiple properties")
        if spouse_owns_pvt:
            reasons.append("Spouse owns private property or multiple properties")
        failed_criteria.append(f"Property ownership criterion failed: {', '.join(reasons)}.")
    else:
        passed_criteria.append(
            f"Property ownership criterion met: Resides in '{flat_type}' and owns no private or multiple properties."
        )

    # Rule 4: Per Capita Household Income (PCHHI) Check (<= $2,300)
    if pchhi <= 2300.0:
        passed_criteria.append(f"PCHHI criterion met: Household PCHHI of ${pchhi:,.2f} is <= $2,300.")
    else:
        failed_criteria.append(f"PCHHI criterion failed: Household PCHHI of ${pchhi:,.2f} exceeds the $2,300 limit.")

    # -------------------------------------------------------------
    # Eligibility & Payout Determination
    # -------------------------------------------------------------
    is_eligible = len(failed_criteria) == 0
    payout_amount = 0.0

    if is_eligible:
        # Payout matrix mapping
        if pchhi <= 1500.0:
            payout_matrix = {
                "1-2 Room": 1080.0,
                "3-Room": 860.0,
                "4-Room": 650.0,
                "5-Room Live-In": 430.0,
            }
        else:  # $1,501 - $2,300
            payout_matrix = {
                "1-2 Room": 540.0,
                "3-Room": 430.0,
                "4-Room": 325.0,
                "5-Room Live-In": 215.0,
            }

        payout_amount = payout_matrix.get(flat_type, 0.0)

    return {
        "is_eligible": is_eligible,
        "payout_amount": payout_amount,
        "pchhi": pchhi,
        "passed_criteria": passed_criteria,
        "failed_criteria": failed_criteria,
    }


if __name__ == "__main__":
    # Quick Test Case 1: Standard Eligible Senior (3-Room, PCHHI <= $1,500)
    test_user_1 = {
        "age": 68,
        "total_cpf_contributions_at_55": 95000.0,
        "is_self_employed_or_platform": False,
        "net_trade_income_avg": 0.0,
        "hdb_flat_type": "3-Room",
        "owns_private_or_multiple_properties": False,
        "spouse_owns_private_or_multiple_properties": False,
        "monthly_household_income": 2000.0,
        "household_members_count": 2,  # PCHHI = $1,000
        "on_comcare_lta": False,
    }

    res1 = evaluate_silver_support_eligibility(test_user_1)
    print("--- Test Case 1 Result ---")
    print(f"Eligible: {res1['is_eligible']} | Quarterly Payout: ${res1['payout_amount']} | PCHHI: ${res1['pchhi']}")

    # Quick Test Case 2: ComCare LTA Override
    test_user_2 = {
        "age": 70,
        "total_cpf_contributions_at_55": 200000.0,  # Exceeds standard limit
        "is_self_employed_or_platform": False,
        "net_trade_income_avg": 0.0,
        "hdb_flat_type": "5-Room Owned / Private",  # Ineligible under standard rules
        "owns_private_or_multiple_properties": False,
        "spouse_owns_private_or_multiple_properties": False,
        "monthly_household_income": 5000.0,
        "household_members_count": 1,
        "on_comcare_lta": True,  # Overrides standard rules
    }

    res2 = evaluate_silver_support_eligibility(test_user_2)
    print("\n--- Test Case 2 Result (ComCare LTA) ---")
    print(f"Eligible: {res2['is_eligible']} | Quarterly Payout: ${res2['payout_amount']}")