"""Dataset license policy, isolated from authoring and release document models."""

import re
from typing import Annotated

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
from pydantic import AfterValidator, Field


def validate_dataset_license(value: str) -> str:
    """Accept one canonical, listed SPDX identifier, or STAC's `other` value.

    Expressions and custom LicenseRef identifiers have no single SPDX license
    page. The authoring model and overview generator check the source URL for
    `other`; legacy published overviews remain readable without that link.
    `other` also preserves ambiguous licensing (for example, LBDA and OWDA);
    its source URL may be a provider study page without explicit license terms.
    The pinned packaging dependency supplies the SPDX list; no network is used.
    """
    if value == "other":
        return value
    message = "license must be a canonical listed SPDX identifier or `other`"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", value) or value.startswith("LicenseRef-"):
        raise ValueError(message)
    try:
        canonical = canonicalize_license_expression(value)
    except InvalidLicenseExpression as exc:
        raise ValueError(message) from exc
    if canonical != value:
        raise ValueError(f"{message}; use {canonical!r}")
    return value


DatasetLicense = Annotated[
    str,
    Field(description="One canonical listed SPDX license identifier, or `other` with a source URL."),
    AfterValidator(validate_dataset_license),
]
