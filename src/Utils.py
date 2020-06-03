from Box2D import b2Body, b2ContactEdge


def checkGroundContact(body):
    """Checks if given body is in contact with ground.
    Ground should be marked by setting userData to 'ground'
    Argument body is of type b2Body
    """
    ceList = body.contacts
    for o in ceList:
        if o.other.userData == 'ground':
            return True
    return False
