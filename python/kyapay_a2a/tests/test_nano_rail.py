# Copyright 2026 Skyfire Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Nano (XNO) settlement rail."""

import pytest

from kyapay_a2a.rails.nano import (
    NanoPaymentResult,
    NanoQuote,
    NanoRail,
    PaymentNotConfirmed,
    create_nano_payment_requirement,
    usd_to_raw,
)

DEST = "nano_1exampleaddress"


def test_nano_quote_is_feeless_and_fast():
    """Nano should quote a zero fee and sub-second finality."""
    rail = NanoRail()
    quote = rail.quote(1.00)
    assert isinstance(quote, NanoQuote)
    assert quote.rail == "nano-xno"
    assert quote.fee_usd == 0.0
    assert quote.finality_s < 1.0


def test_unconfigured_rail_fails_closed():
    """A NanoRail without an rpc must never report a payment as settled."""
    rail = NanoRail()
    with pytest.raises(PaymentNotConfirmed):
        rail.verify(
            "deadbeef",
            destination=DEST,
            amount_raw="1000000000000000000000000",
        )
    with pytest.raises(PaymentNotConfirmed):
        # No block hash -> nothing could have been verified as settled.
        rail.pay(destination=DEST, amount_raw="1000000000000000000000000")


def test_verify_confirms_a_matching_block():
    """A confirmed block for the exact amount/destination settles."""
    amount = "1000000000000000000000000"

    def fake_rpc(request):
        assert request["action"] == "block_info"
        return {
            "contents": {
                "type": "send",
                "account": "nano_1buyer",
                "amount": amount,
                "link_as_account": DEST,
            },
            "confirmed": True,
        }

    rail = NanoRail(rpc=fake_rpc)
    result = rail.verify(
        "deadbeef",
        destination=DEST,
        amount_raw=amount,
    )
    assert isinstance(result, NanoPaymentResult)
    assert result.settled is True
    assert result.rail == "nano-xno"
    assert result.block_hash == "deadbeef"
    assert result.amount_raw == amount


def test_amount_mismatch_fails_closed():
    """A block whose amount differs from the advertised price must not settle."""
    def fake_rpc(request):
        return {
            "contents": {
                "type": "send",
                "amount": "1000000000000000000000000",  # advertised
            },
        }

    rail = NanoRail(rpc=fake_rpc)
    with pytest.raises(PaymentNotConfirmed):
        rail.verify(
            "deadbeef",
            destination=DEST,
            amount_raw="500000000000000000000000",  # underpaid
        )


def test_destination_mismatch_fails_closed():
    """A block sent to the wrong address must not settle."""
    def fake_rpc(request):
        return {
            "contents": {
                "type": "send",
                "amount": "1000000000000000000000000",
                "link_as_account": "nano_1someoneelse",
            },
        }

    rail = NanoRail(rpc=fake_rpc)
    with pytest.raises(PaymentNotConfirmed):
        rail.verify(
            "deadbeef",
            destination=DEST,
            amount_raw="1000000000000000000000000",
        )


def test_pay_requires_a_block_hash():
    """pay() with no buyer block hash fails closed rather than settle."""
    def fake_rpc(request):
        return {"contents": {"type": "send", "amount": "1"}}

    rail = NanoRail(rpc=fake_rpc)
    with pytest.raises(PaymentNotConfirmed):
        rail.pay(destination=DEST, amount_raw="1")


def test_usd_to_raw_conversion():
    """USD converts to an exact raw amount for the XNO/USD rate."""
    # $1.00 at XNO=$1.00 is exactly 10^30 raw.
    assert usd_to_raw("1.00", "1.0") == str(10**30)
    # $0.01 at XNO=$1.00 is 10^28 raw.
    assert usd_to_raw("0.01", "1.0") == str(10**28)


def test_create_nano_payment_requirement_binds_amount():
    """The requirement should carry the exact raw amount for the price."""
    requirement = create_nano_payment_requirement(
        price_usd="0.01",
        resource="/api/service",
        nano_address=DEST,
        xno_usd="1.0",
    )
    assert requirement["rail"] == "nano-xno"
    assert requirement["price_usd"] == "0.01"
    assert requirement["amount_raw"] == str(10**28)
    assert requirement["nano_address"] == DEST
    assert requirement["resource"] == "/api/service"
    assert requirement["expires_at"] is None
