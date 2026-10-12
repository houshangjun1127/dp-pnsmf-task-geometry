"""Account only events actually recorded by the prototype."""
import math
from dp_accounting import dp_event, rdp
ORDERS=list(range(2,65))+[128.,256.]

def account(events,delta=1e-5):
    if any(not e['protected'] or e['gaussian_multiplier']<=0 for e in events):
        return {'private':False,'epsilon_library':None,'reason':'Contains unprotected event'}
    a=rdp.RdpAccountant(ORDERS)
    rho=0.
    for e in events:
        a.compose(dp_event.GaussianDpEvent(e['gaussian_multiplier']),count=e['repetitions'])
        rho+=e['repetitions']/(2*e['gaussian_multiplier']**2)
    plain=min(alpha*rho+math.log(1/delta)/(alpha-1) for alpha in ORDERS)
    return dict(private=True,rho_coefficient=rho,epsilon_plain_discrete=plain,
        epsilon_plain_continuous=rho+2*math.sqrt(rho*math.log(1/delta)),
        epsilon_library=float(a.get_epsilon(delta)),delta=delta,orders=ORDERS)

def schedule(k,T,count_sigma,sigmas):
    return [dict(name='count_'+str(i),gaussian_multiplier=count_sigma,protected=True,repetitions=1)
            for i in [1,2]]+[
            dict(name=name,gaussian_multiplier=s/math.sqrt(weight),protected=True,repetitions=T)
            for name,weight,s in zip(['local_gram','local_rhs','shared_gram'],[k,k,1],sigmas)]

def calibrate(k,T,target=2.193078238018666,delta=1e-5):
    # One shared normalized base scale, independently checked in tests.
    lo,hi=.001,10000.
    for _ in range(90):
        mid=(lo+hi)/2
        if account(schedule(k,T,mid,[mid]*3),delta)['epsilon_library']>target: lo=mid
        else: hi=mid
    return hi
