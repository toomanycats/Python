"""This is a snowball sampler for Twitter data."""

import networkx as nx
import time
import os
import tweepy
import ipdb
from pymongo import MongoClient
import ConfigParser


enc = lambda x: x.encode('ascii', errors='ignore')

def get_api():
    tokens = Tokens()
    auth = tweepy.OAuthHandler(tokens.consumer_key,
                               tokens.consumer_secret
                               )

    auth.set_access_token(tokens.access_token,
                          tokens.access_token_secret
                          )

    api = tweepy.API(auth)

    return api


def wait_for_limit(error):
    try:
        if error[0][0]['message'] == 'Rate limit exceeded':
            print 'Rate limited. Sleeping for 15 minutes.'
            time.sleep(15 * 60 + 15)
            return "slept"

    except:
        print str(error)
        return None


class Tokens(object):
    def __init__(self):
        config_file = "/home/daniel/git/Python2.7/DataScience/twitter_tokens.cfg"
        config_parser = ConfigParser.RawConfigParser()
        config_parser.read(config_file)
        self.consumer_key = config_parser.get("tokens", "consumer_key")
        self.consumer_secret = config_parser.get("tokens", "consumer_secret")
        self.access_token = config_parser.get("tokens", "access_token")
        self.access_token_secret = config_parser.get("tokens", "access_token_secret")


class DB(object):
    """Simple data base class to open mongoDB"""
    def __init__(self):
        client = MongoClient('localhost', 27017)
        db = client.snowball
        self.nodes = db.wfm_centers
        self.edges = db.wfm_followers
        self.content = db.content


class CursorLoop(object):
    def __init__(self, cursor):
        self.cursor = cursor

    def loop_gen(self):
        while True:
            try:
                item = self.cursor.next()
                yield(item)

            except StopIteration:
                break

            except tweepy.TweepError, error:
                mess = wait_for_limit(error)
                if mess == 'slept':
                    item = self.cursor.next()
                    yield(item)
                else:
                    continue

    def loop_list(self):
        Item = []
        while True:
            try:
                item = self.cursor.next()
                Item.append(item)

            except StopIteration:
                break

            except tweepy.TweepError, error:
                mess = wait_for_limit(error)
                if mess == 'slept':
                    item = self.cursor.next()
                else:
                    continue
        return Item


class Sample(object):
    def __init__(self, api, seed, max_followers=20, max_depth=5):
        db = DB()
        self.nodes = db.nodes
        self.edges = db.edges

        self.seed = seed
        self.max_followers = max_followers
        self.max_depth = max_depth
        self.api = api

    def main(self):
        screen_name = self.seed
        self.get_screen_name_info(screen_name, depth=0)

    def _get_list_followers(self, user_id):
        try:
            c = tweepy.Cursor(self.api.followers, id=user_id).items()

        except tweepy.TweepError, error:
            print error

            mess = wait_for_limit(error)
            if mess == "slept":
                self._get_list_followers(user_id)

        except Exception, err:
            print err
            return None

        return c

    def _get_user(self, screen_name):

        try:
            user = self.api.get_user(screen_name)
            return user

        except tweepy.TweepError, error:
            print error

            mess = wait_for_limit(error)
            if mess == "slept":
                self._get_user(screen_name)

        except Exception, err:
            print err
            return None

    def _make_user_dict(self, screen_name, depth):
        user_info = self._get_user(screen_name)

        try:# follwers ids can fail for permission error
            user = {'name': enc(user_info.name),
                    'screen_name': enc(user_info.screen_name),
                    'id': user_info.id,
                    'friends_count': user_info.friends_count,
                    'followers_count': user_info.followers_count,
                    'followers_ids': user_info.followers_ids()
                   }

            return user, depth

        except tweepy.TweepError, error:
            mess = wait_for_limit(error)
            if mess == "slept":
                user, depth = self._make_user_dict(screen_name, depth)
                return user, depth
            else:
                depth -= 1
                return None, depth

        except Exception, error:
            print error
            return None, depth

    def get_screen_name_info(self, screen_name, depth):
        print '\nRetrieving user details for twitter id %s\n' % enc(screen_name)

        cursor = self.nodes.find({screen_name:{"$exists":True}})
        if cursor.count() > 0:
            user = cursor.next()
            depth = self.get_followers(user, depth)
            return depth

        else:
            user, depth = self._make_user_dict(screen_name, depth)
            if user is not None:
                self.nodes.insert(user)
                depth = self.get_followers(user, depth)
                return depth

            else:
                return depth

    def get_followers(self, user, depth):
        cursor = self.edges.find({user['screen_name']:{"$exists":True}})
        if user['followers_count'] > 0 and not cursor.count() > 0:
            print 'Retrieving friends for user: %s' %user['screen_name']
            Id = self.lookup_followers(user['id'], user['screen_name'])
            cursor = self.edges.find({'_id':Id})
            followers = cursor.next()

        else:
            return depth

        if depth < self.max_depth + 1:
            for follower in followers['followers']:
                fscreen_name = follower['screen_name']
                depth += 1
                print "depth: %i" %depth
                depth = self.get_screen_name_info(fscreen_name, depth)

        return depth

    def lookup_followers(self, user_id, screen_name):
        cursor = self._get_list_followers(user_id)

        if cursor is None:
            return None

        data = []
        cnt = 0
        while True:
            try:
                follower = cursor.next()

            except tweepy.TweepError, error:
                mess = wait_for_limit(error)
                if mess == 'slept':
                    follower = cursor.next()
                else:
                    continue
            except StopIteration:
                break

            cnt += 1
            new_data = {'follower_id':follower.id,
                        'screen_name':follower.screen_name,
                        'name':follower.name
                        }

            data.append(new_data)

            print "follower:%s" % new_data['screen_name']

            if cnt == self.max_followers:
                break

        followers_container = {'node':screen_name, 'followers':data}
        Id = self.edges.insert(followers_container)
        return Id


class Edges(object):
    def __init__(self, seed=None):
        db = DB()
        self.nodes = db.nodes
        self.edges = db.edges

        self.users = {}
        self.edges_list = []
        self.black_list = []
        self.seed = seed

    def load_users(self):
        cursor = self.nodes.find({}, {"screen_name": True,
                                      "followers_count": True,
                                      "_id": False
                                      }
                                )
        while True:
            try:
                item = cursor.next()
                screen_name = item['screen_name']
                num_followers = item['followers_count']
                self.users[screen_name] = {'num_followers': num_followers}

            except StopIteration:
                break

            except KeyError:
               print "no screen names found in DB"
               raise Exception

    def _unpack_followers(self, cursor):
        item = cursor.next()
        followers = item['followers']
        follower_screen_names = map(lambda x:x['screen_name'], followers)

        return follower_screen_names

    def get_followers(self, screen_name):
        cursor = self.edges.find({"node":screen_name})
        if cursor.count() > 0:
            followers = self._unpack_followers(cursor)
            return followers

        else:
            return None

    def build_edges(self, screen_name):
        if screen_name in self.black_list:
            return
        else:
            self.black_list.append(screen_name)

        followers = self.get_followers(screen_name)

        if followers is None or len(followers) < 2:
            return

        data = self._build_edge(screen_name, followers)

        if data is not None:
            self.edges_list += (data)
        else:
            return

        # if sample was interupted, the data may be incomplete
        # and cause a failure here.
        try:
            for screen_name in followers:
                self.build_edges(screen_name)
        except ValueError:
            return

    def _build_edge(self, screen_name, followers):
        new_data = []
        try: # sometimes the prog quit early and data is incomplete
            for follower_screen_name in followers:
                weight = self.users[follower_screen_name]['num_followers']
                new_data.append([screen_name, follower_screen_name, weight])

        except KeyError:
            weight = 1

        return new_data

    def main(self):
        self.load_users()
        self.build_edges(self.seed)

        return self.edges_list


class Network(object):
    def __init__(self, seed, outfile):
        self.di_graph = nx.DiGraph()
        self.weights = {}
        self.seed = seed
        root = '/home/daniel/git/Python2.7/DataScience'
        self.graphml_path = os.path.join(root, outfile)

    def load_network(self, network):
        for screen_name, followed_by, weight in network:
            weight = int(weight)
            self.di_graph.add_edge(screen_name, followed_by, weight=weight)
            self.weights[screen_name] = weight

    def main(self):
        network = Edges(self.seed).main()
        self.load_network(network)

        f = open(self.graphml_path, 'w')
        nx.GraphMLWriter(self.di_graph).dump(f)
        f.close()


class TwitterContent(object):
    def __init__(self, api, num_tweets):
        self.db = DB()
        self.nodes = self.db.nodes
        self.content = self.db.content
        self.num = num_tweets
        self.api = api

    def get_name_list(self):
        cursor = self.nodes.find({}, {"screen_name":True, "_id":False})
        name_list = CursorLoop(cursor).loop_list()
        name_list = map(lambda d: d['screen_name'], name_list)

        return name_list

    def get_user_tweets(self, screen_name):
        cursor = tweepy.Cursor(self.api.user_timeline, id=screen_name)
        time_line = cursor.items(self.num)

        out = []
        for item in time_line:
            out.append(item.text)

        return out

    def store_tweet_in_db(self, screen_name):
        try:
            tweets = self.get_user_tweets(screen_name)
            data = {screen_name : tweets}
            self.content.insert(data)

        except tweepy.TweepError, error:
            mess = wait_for_limit(error)
            if mess == 'slept':
                self.store_tweet_in_db(screen_name)
            else:
                return

    def main(self):
        name_list = self.get_name_list()

        for name in name_list:
            self.store_tweet_in_db(name)



